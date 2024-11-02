import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

# Experience replay buffer size
REPLAY_SIZE = 1048576
# Minibatch size
SMALL_BATCH_SIZE = 64
BIG_BATCH_SIZE = 128
BATCH_SIZE_DOOR = 1000

# Hyperparameters for Dueling DQN
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EPSILON_DECAY = 50000
LR = 0.001
ALPHA = 0.5
BETA_START = 0.4
BETA_FRAMES = 100000

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DuelingDQN, self).__init__()
        self.action_space = action_space

        # Using ResNet101 for feature extraction
        self.feature_extractor = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.feature_extractor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal_(self.feature_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.feature_extractor.fc = nn.Identity()

        self.value_stream = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )

        # Apply Xavier initialization to the value and advantage streams
        self.value_stream.apply(self._init_weights)
        self.advantage_stream.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Custom weight initialization using Xavier for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if self.tree[left] >= s:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = ALPHA

        self.max_priority = 1.0
        self.min_priority = 1.0

    def add(self, error, sample):
        p = (error + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, p)
        self.min_priority = min(self.min_priority, p)
        self.tree.add(p, sample)

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idxs, errors):
        """批量更新多个优先级"""
        if not hasattr(idxs, '__iter__') or not hasattr(errors, '__iter__'):
            raise TypeError("idxs and errors must be iterable")
        if len(idxs) != len(errors):
            raise ValueError("idxs and errors must have the same length")

        for idx, error in zip(idxs, errors):
            p = (error + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self.min_priority = min(self.min_priority, p)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.write if self.tree.write != 0 else self.capacity


class DQNAgent:
    def __init__(self, input_channels, action_space, model_file, model_folder):
        self.global_step = 0
        self.train_step = 0
        self.state_dim = input_channels
        self.action_space = action_space
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
        self.eval_net = DuelingDQN(input_channels, action_space).to(device)
        self.target_net = DuelingDQN(input_channels, action_space).to(device)
        self.update_target_network()
        trainable_params = filter(lambda p: p.requires_grad, self.eval_net.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=15000, gamma=0.8)

        self.epsilon = INITIAL_EPSILON
        self.beta = BETA_START
        self.model_folder = model_folder
        self.model_file = model_file
        self.writer = SummaryWriter(log_dir=f'./logs/run_{self.global_step}')
        self.scaler = GradScaler()

        # Load checkpoint or DQN
        self.load_checkpoint_or_model()

    def load_checkpoint_or_model(self):
        checkpoint_dir = self.model_folder
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith('.pth')],
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                reverse=True
            )
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Loading checkpoint from {checkpoint_path}...\n")

                checkpoint = torch.load(checkpoint_path, map_location=device)

                self.eval_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])  # Ensure target_net is synced
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.global_step = checkpoint.get('global_step', 0)
                self.train_step = checkpoint.get('train_step', 0)
                self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
                self.beta = checkpoint.get('beta', BETA_START)
                print(f"Checkpoint loaded successfully. Global Step: {self.global_step}, Train Step: {self.train_step}")
                return  # Successfully loaded from checkpoint

        # If no checkpoints found, try to load from model_file
        print("No checkpoints found. Attempting to load model from file...\n")
        self.load_model()

    def load_model(self):
        """Load the model and optimizer state from a file."""
        if os.path.exists(self.model_file):
            try:
                # Handle the FutureWarning by setting weights_only=True if appropriate
                # However, since you're loading a full checkpoint (including optimizer and scheduler),
                # you need weights_only=False. Be aware of the warning for future updates.
                checkpoint = torch.load(self.model_file, map_location=device)
                self.eval_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])  # Ensure target_net is synced
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
                self.beta = checkpoint.get('beta', BETA_START)
                self.global_step = checkpoint.get('global_step', 0)
                self.train_step = checkpoint.get('train_step', 0)
                print(f"Model loaded successfully from {self.model_file}.")
            except Exception as e:
                print(f"Failed to load model from {self.model_file}: {e}")
                print("Initializing networks randomly.")
                self.initialize_networks()
        else:
            print(f"Model file {self.model_file} does not exist. Initializing networks randomly.")
            self.initialize_networks()

    def initialize_networks(self):
        """Initialize networks with random weights."""
        print("Initializing evaluation and target networks with random weights.")
        self.eval_net.apply(self._init_weights)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        # Reset optimizer and scheduler
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.eval_net.parameters()), lr=LR,
                                    weight_decay=1e-5)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.1)
        self.epsilon = INITIAL_EPSILON
        self.beta = BETA_START
        self.global_step = 0
        self.train_step = 0

    @staticmethod
    def _init_weights(m):
        """Custom weight initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def update_target_network(self, train_step_interval=5000):
        if self.train_step % train_step_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print(f"Target network updated at train step {self.train_step}")

    def choose_action(self, state, action_mask):
        if random.random() <= self.epsilon:
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = None
        else:
            state = state.unsqueeze(0).to(device)
            if len(state.shape) == 4:
                q_values = self.eval_net(state).squeeze(0)
                action_mask_tensor = torch.FloatTensor(action_mask).to(device)
                masked_q_values = q_values * action_mask_tensor + (1 - action_mask_tensor) * (-1e9)
                action = torch.argmax(masked_q_values).item()
            else:
                raise ValueError("State input must have 4 dimensions: [batch, channels, height, width]")
        self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        state = state.to(device)
        next_state = next_state.to(device)
        max_priority = self.replay_buffer.max_priority
        self.replay_buffer.add(max_priority, (state, action, reward, next_state, done))
        self.global_step += 1

    def log_metrics(self, loss, reward_sum, q_max, q_min, q_mean, target_q_max, target_q_min, target_q_mean,
                    total_norm):
        self.writer.add_scalar('Loss/train', loss, self.global_step)
        self.writer.add_scalar('Epsilon', self.epsilon, self.global_step)
        self.writer.add_scalar('Beta', self.beta, self.global_step)
        self.writer.add_scalar('Total reward', reward_sum, self.global_step)

        self.writer.add_scalar('Q-values/max', q_max, self.global_step)
        self.writer.add_scalar('Q-values/min', q_min, self.global_step)
        self.writer.add_scalar('Q-values/mean', q_mean, self.global_step)

        self.writer.add_scalar('Target Q-values/max', target_q_max, self.global_step)
        self.writer.add_scalar('Target Q-values/min', target_q_min, self.global_step)
        self.writer.add_scalar('Target Q-values/mean', target_q_mean, self.global_step)

        self.writer.add_scalar('Gradient norm', total_norm, self.global_step)

        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning Rate', current_lr, self.global_step)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # 更新 Beta 值，用于优先经验回放
        self.beta = min(1.0, self.beta + (1.0 - BETA_START) / BETA_FRAMES)

        # 从回放缓冲区采样
        samples, idxs, is_weights = self.replay_buffer.sample(batch_size, self.beta)
        batch = list(zip(*samples))
        state_batch = torch.stack(batch[0]).to(device, non_blocking=True)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_state_batch = torch.stack(batch[3]).to(device, non_blocking=True)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=device)

        with autocast(device_type=device.type):
            # 当前 Q 值
            q_values = self.eval_net(state_batch).gather(1, action_batch).squeeze(1)

            # 计算目标 Q 值
            with torch.no_grad():
                # Double DQN: 动作选择由 eval_net 完成，Q 值由 target_net 计算
                next_actions = self.eval_net(next_state_batch).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

            # TD 误差
            td_errors = q_values - target_q_values

            # 使用重要性采样权重计算损失
            loss = (is_weights * td_errors.pow(2)).mean()

        # 反向传播和优化
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 更新学习率调度器
        self.scheduler.step()

        # 计算总奖励
        reward_sum = torch.sum(reward_batch).item()

        # 计算 Q 值的最大值、最小值和平均值
        q_max = q_values.max().item()
        q_min = q_values.min().item()
        q_mean = q_values.mean().item()

        # 计算 Target Q 值的最大值、最小值和平均值
        target_q_max = target_q_values.max().item()
        target_q_min = target_q_values.min().item()
        target_q_mean = target_q_values.mean().item()

        # 日志记录
        self.log_metrics(
            loss.item(),
            reward_sum,
            q_max,
            q_min,
            q_mean,
            target_q_max,
            target_q_min,
            target_q_mean,
            total_norm
        )

        # 计算并更新优先经验回放中的优先级
        td_errors_cpu = td_errors.detach().cpu().numpy()
        abs_td_errors = np.abs(td_errors_cpu)
        self.replay_buffer.update(idxs, abs_td_errors)

        # 增加训练步数
        self.train_step += 1
        self.global_step += 1  # 增加 global_step

        # 定期更新目标网络
        self.update_target_network()

        # 定期保存模型检查点
        self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.model_folder, f"checkpoint_step_{self.global_step}.pth")
        torch.save({
            'global_step': self.global_step,
            'train_step': self.train_step,
            'model_state_dict': self.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
        }, checkpoint_path)
        print(f"Checkpoint saved at step {self.global_step} to {checkpoint_path}")

        with open(os.path.join(self.model_folder, "last_step.txt"), "w") as f:
            f.write(str(self.global_step))

        checkpoints = [f for f in os.listdir(self.model_folder) if
                       f.startswith("checkpoint_step_") and f.endswith('.pth')]
        if len(checkpoints) > 20:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_folder, x)))
            oldest_checkpoint = checkpoints[0]
            os.remove(os.path.join(self.model_folder, oldest_checkpoint))
            print(f"Deleted oldest checkpoint: {oldest_checkpoint}")

    def save_model(self, episode):
        """Save the model and optimizer state at a given episode."""
        save_path = f'{self.model_folder}/dueling_dqn_episode_{episode}.pth'
        torch.save({
            'model_state_dict': self.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
        }, save_path)
        print(f"Model saved at episode {episode} to {save_path}")
