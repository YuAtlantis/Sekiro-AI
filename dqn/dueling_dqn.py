import gzip
import os
import pickle
import random
import torch
import time
import math
import threading
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from threading import Lock, Event
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

# Experience replay buffer size
REPLAY_SIZE = 7000
# Minibatch size
SMALL_BATCH_SIZE = 64
BIG_BATCH_SIZE = 128
BATCH_SIZE_DOOR = 1000

# Hyperparameters for Dueling DQN
GAMMA = 0.99
INITIAL_EPSILON = 0.8
FINAL_EPSILON = 0.01
EPSILON_DECAY = 120
LR = 1e-5
ALPHA = 0.7
BETA_START = 0.4
BETA_FRAMES = 2000

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DuelingDQN, self).__init__()
        self.action_space = action_space

        # Use ResNet50 as the feature extractor
        self.resnet = resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

        # Get the output feature dimension of ResNet50
        flattened_size = 2048  # ResNet50 last layer output dimension

        # Transformer parameters
        self.embed_dim = 256  # Embedding dimension
        self.seq_length = flattened_size // self.embed_dim  # Sequence length (2048 / 256 = 8)

        # Adjust feature dimensions to fit the Transformer
        self.fc = nn.Linear(flattened_size, self.seq_length * self.embed_dim)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_length, self.embed_dim))

        # Multi-head self-attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1,
                                                     batch_first=True)

        # Value stream and advantage stream with Dropout
        self.value_stream = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        # ResNet feature extraction
        x = self.resnet(x)  # Output shape: [batch_size, 2048]

        x = self.fc(x)  # Adjust to [batch_size, seq_length * embed_dim]
        x = x.view(x.size(0), self.seq_length, self.embed_dim)  # Reshape to [batch_size, seq_length, embed_dim]

        # Add positional encoding
        x = x + self.positional_encoding  # [batch_size, seq_length, embed_dim]

        # Self-Attention
        x, _ = self.attention_layer(x, x, x)  # Output shape: [batch_size, seq_length, embed_dim]

        x = x.contiguous().view(x.size(0), -1)  # Flatten to [batch_size, seq_length * embed_dim]

        # Compute value function and advantage function
        v = self.value_stream(x)
        a = self.advantage_stream(x)

        # Combine to form Q-values
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
        self.size = 0
        self.lock = Lock()
        print("PrioritizedReplayBuffer initialized with lock.")

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = Lock()

    def add(self, error, sample):
        state, action, reward, next_state, done = sample
        if torch.isnan(state).any() or torch.isnan(next_state).any() or math.isnan(reward):
            print("NaN detected in sample, skipping.")
            return
        p = (error + 1e-5) ** self.alpha
        with self.lock:
            self.max_priority = max(self.max_priority, p)
            self.min_priority = min(self.min_priority, p)
            self.tree.add(p, sample)
            if self.size < self.capacity:
                self.size += 1

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
        """Batch update multiple priorities"""
        if not hasattr(idxs, '__iter__') or not hasattr(errors, '__iter__'):
            raise TypeError("idxs and errors must be iterable")
        if len(idxs) != len(errors):
            raise ValueError("idxs and errors must have the same length")

        with self.lock:
            for idx, error in zip(idxs, errors):
                p = (error + 1e-5) ** self.alpha
                self.max_priority = max(self.max_priority, p)
                self.min_priority = min(self.min_priority, p)
                self.tree.update(idx, p)

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, input_channels, action_space, model_file, model_folder):
        self.global_step = 0
        self.global_episode = 0

        self.state_dim = input_channels
        self.action_space = action_space
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
        self.eval_net = DuelingDQN(input_channels, action_space).to(device)
        self.target_net = DuelingDQN(input_channels, action_space).to(device)
        self.update_target_network()
        trainable_params = filter(lambda p: p.requires_grad, self.eval_net.parameters())
        self.optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-5)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100000)

        self.epsilon = INITIAL_EPSILON
        self.beta = BETA_START
        self.model_folder = model_folder
        self.model_file = model_file
        self.scaler = GradScaler()

        self.save_lock = Lock()

        # Added: Initialize best reward
        self.best_reward = -float('inf')

        # Training thread control
        self.training_thread = None
        self.training_stop_event = Event()

        # Load checkpoint or replay_buffer
        self.load_largest_replay_buffer()
        self.load_checkpoint_or_model()

        self.writer = SummaryWriter(log_dir='./logs')

        # Start training thread
        self.start_training_thread()

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
        self.best_reward = -float('inf')  # Reset best reward

    @staticmethod
    def _init_weights(m):
        """Custom weight initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def update_target_network(self):
        if self.global_step < 50:
            update_interval = 6
        else:
            update_interval = 18
        if self.global_step % update_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print(f"Target network updated at step {self.global_step}")

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
        self.epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * math.exp(-1. * self.global_step / EPSILON_DECAY)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(self.replay_buffer.max_priority, (state, action, reward, next_state, done))

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

        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning Rate', current_lr, self.global_step)

    def train_step(self):
        """Perform a single training step."""
        # Update Beta value for prioritized experience replay
        self.beta = min(1.0, self.beta + (1.0 - BETA_START) / BETA_FRAMES)

        # Sample from replay buffer
        samples, idxs, is_weights = self.replay_buffer.sample(BIG_BATCH_SIZE, self.beta)
        batch = list(zip(*samples))
        state_batch = torch.stack(batch[0]).to(device, non_blocking=True)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_state_batch = torch.stack(batch[3]).to(device, non_blocking=True)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=device)

        with autocast(device_type=device.type):
            # Current Q values
            q_values = self.eval_net(state_batch).gather(1, action_batch).squeeze(1)

            # Calculate target Q values
            with torch.no_grad():
                # Double DQN: Action selection by eval_net, Q values by target_net
                next_actions = self.eval_net(next_state_batch).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

            # TD errors
            td_errors = q_values - target_q_values

            # Compute loss using importance sampling weights
            loss = (is_weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update learning rate scheduler
        self.scheduler.step()

        # Compute total reward
        reward_sum = torch.sum(reward_batch).item()

        # Compute Q value statistics
        q_max = q_values.max().item()
        q_min = q_values.min().item()
        q_mean = q_values.mean().item()

        # Compute target Q value statistics
        target_q_max = target_q_values.max().item()
        target_q_min = target_q_values.min().item()
        target_q_mean = target_q_values.mean().item()

        # Log metrics
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

        # Check and save best model
        self.check_and_save_best_model(reward_sum)

        # Calculate and update priorities in prioritized experience replay
        td_errors_cpu = td_errors.detach().cpu().numpy()
        abs_td_errors = np.abs(td_errors_cpu)
        self.replay_buffer.update(idxs, abs_td_errors)

        # Periodically update target network
        self.update_target_network()

        # Periodically save model checkpoints
        self.save_checkpoint()

        # Increment global_step
        self.global_step += 1

        # Save replay buffer every 10 steps
        if self.global_step % 10 == 0:
            replay_buffer_path = os.path.join(self.model_folder, f"replay_buffer_size_{len(self.replay_buffer)}.pkl.gz")
            self.save_replay_buffer_async(replay_buffer_path)

    def training_loop(self):
        """Continuous training loop running in a separate thread."""
        while not self.training_stop_event.is_set():
            buffer_length = len(self.replay_buffer)
            if buffer_length >= 3500:
                print(f"Starting training step with buffer size: {buffer_length}")
                self.train_step()
                print(f"Completed training step. Current step: {self.global_step}")
                time.sleep(60)
            else:
                print(f"Current buffer size is: {buffer_length}")
                time.sleep(5)

    def start_training_thread(self):
        """Start the training thread."""
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()

    def stop_training_thread(self):
        """Stop the training thread."""
        self.training_stop_event.set()
        if self.training_thread is not None:
            self.training_thread.join()
            print("Training thread stopped.")

    def check_and_save_best_model(self, reward_sum):
        """
        Check if the current reward exceeds the best reward, and if so, save the model as the best model.
        """
        if reward_sum > self.best_reward:
            print(f"New best reward: {reward_sum} (previous best: {self.best_reward})")
            self.best_reward = reward_sum
            self.save_best_model()

    def save_replay_buffer(self, path):
        """Save the replay buffer to a compressed file."""
        temp_path = f"{path}.tmp"
        with self.save_lock:
            try:
                with gzip.open(temp_path, 'wb') as f:
                    pickle.dump(self.replay_buffer, f)
                os.replace(temp_path, path)
                print(f"Replay buffer saved to {path}")
            except Exception as e:
                print(f"Failed to save replay buffer to {path}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def save_replay_buffer_async(self, path):
        """Asynchronously save the replay buffer."""
        thread = threading.Thread(target=self.save_replay_buffer, args=(path,))
        thread.start()

    def save_checkpoint(self):
        # Save the model checkpoint
        checkpoint_path = os.path.join(self.model_folder, f"checkpoint_step_{self.global_step}.pth")
        torch.save({
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'model_state_dict': self.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'best_reward': self.best_reward,
        }, checkpoint_path)
        print(f"Checkpoint saved at step {self.global_step} to {checkpoint_path}")

        with open(os.path.join(self.model_folder, "last_step.txt"), "w") as f:
            f.write(str(self.global_step))

        # Manage old checkpoints and replay buffer files
        self.manage_old_checkpoints()

    def load_checkpoint_or_model(self):
        # Load the model checkpoint
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
                self.global_episode = checkpoint.get('global_episode', 0)
                self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
                self.beta = checkpoint.get('beta', BETA_START)
                self.best_reward = checkpoint.get('best_reward', -float('inf'))  # Restore best reward

                print(
                    f"Checkpoint loaded successfully. Global Step: {self.global_step}, Best Reward: {self.best_reward}")
                return  # Successfully loaded from checkpoint

        # If no checkpoints found, try to load from model_file
        print("No checkpoints found. Attempting to load model from file...\n")
        self.load_model()

    def load_model(self):
        """Load the model and optimizer state from a file."""
        if os.path.exists(self.model_file):
            try:
                checkpoint = torch.load(self.model_file, map_location=device)
                self.eval_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])  # Ensure target_net is synced
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epsilon = checkpoint.get('epsilon', INITIAL_EPSILON)
                self.beta = checkpoint.get('beta', BETA_START)
                self.global_step = checkpoint.get('global_step', 0)
                self.global_episode = checkpoint.get('global_episode', 0)
                self.best_reward = checkpoint.get('best_reward', -float('inf'))  # Restore best reward

                print(f"Model loaded successfully from {self.model_file}. Best Reward: {self.best_reward}")
            except Exception as e:
                print(f"Failed to load model from {self.model_file}: {e}")
                print("Initializing networks randomly.")
                self.initialize_networks()
        else:
            print(f"Model file {self.model_file} does not exist. Initializing networks randomly.")
            self.initialize_networks()

    def load_largest_replay_buffer(self):
        """Load the replay buffer with the largest size from the model folder."""
        replay_buffer_files = [
            f for f in os.listdir(self.model_folder)
            if f.startswith('replay_buffer_size_') and f.endswith('.pkl.gz')
        ]
        if replay_buffer_files:
            # Extract sizes from filenames
            sizes_and_files = []
            for filename in replay_buffer_files:
                try:
                    size_str = filename[len('replay_buffer_size_'):-len('.pkl.gz')]
                    size = int(size_str)
                    sizes_and_files.append((size, filename))
                except ValueError:
                    pass
            if sizes_and_files:
                # Get the file with the largest size
                sizes_and_files.sort(reverse=True)
                largest_size, largest_file = sizes_and_files[0]
                replay_buffer_path = os.path.join(self.model_folder, largest_file)
                print(f"Loading replay buffer from {replay_buffer_path} with size {largest_size}...")
                self.load_replay_buffer(replay_buffer_path)
            else:
                print("No valid replay buffer files found. Starting with an empty replay buffer.")
                self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
        else:
            print("No replay buffer files found. Starting with an empty replay buffer.")
            self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)

    def save_best_model(self):
        """
        Save the current best model. You can choose a dedicated filename, such as 'best_model.pth'.
        """
        best_model_path = os.path.join(self.model_folder, "best_model.pth")
        torch.save({
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'model_state_dict': self.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'best_reward': self.best_reward,
        }, best_model_path)
        print(f"Best model saved with reward {self.best_reward} at {best_model_path}")

    def manage_old_checkpoints(self, max_checkpoints=8):
        checkpoints = [f for f in os.listdir(self.model_folder) if
                       f.startswith("checkpoint_step_") and f.endswith('.pth')]

        if len(checkpoints) > max_checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_folder, x)))
            for checkpoint in checkpoints[:-max_checkpoints]:
                checkpoint_path = os.path.join(self.model_folder, checkpoint)
                os.remove(checkpoint_path)
                print(f"Deleted old checkpoint: {checkpoint_path}")

        replay_buffers = [f for f in os.listdir(self.model_folder) if
                          f.startswith("replay_buffer_size_") and f.endswith('.pkl.gz')]

        valid_replay_buffers = []
        for f in replay_buffers:
            try:
                size_part = f.split('_')[3]
                size = int(size_part.split('.')[0])
                valid_replay_buffers.append((f, size))
            except (IndexError, ValueError):
                print(f"Invalid replay buffer format: {f}")

        valid_replay_buffers.sort(key=lambda x: (x[1], os.path.getmtime(os.path.join(self.model_folder, x[0]))))

        for rb_file, _ in valid_replay_buffers[:-2]:
            rb_file_path = os.path.join(self.model_folder, rb_file)
            os.remove(rb_file_path)
            print(f"Deleted old replay buffer: {rb_file_path}")

    def load_replay_buffer(self, path):
        """Load the replay buffer from a compressed file."""
        if os.path.exists(path):
            try:
                with gzip.open(path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                print(f"Replay buffer loaded from {path}")
            except Exception as e:
                print(f"Failed to load replay buffer from {path}: {e}")
                print("Starting with an empty replay buffer.")
                self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
        else:
            print(f"Replay buffer file {path} does not exist. Starting with an empty replay buffer.")
            self.replay_buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
