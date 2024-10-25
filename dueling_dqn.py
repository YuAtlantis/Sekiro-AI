import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torch.nn.functional as functional
from collections import deque
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

# Experience replay buffer size
REPLAY_SIZE = 2000
# Minibatch size
SMALL_BATCH_SIZE = 16
BIG_BATCH_SIZE = 128
BATCH_SIZE_DOOR = 1000

# Hyperparameters for Dueling DQN
GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
LR = 0.001

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DuelingDQN, self).__init__()
        self.action_space = action_space

        # Using ResNet for feature extraction
        self.feature_extractor = models.resnet152(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer

        # Fully connected layers for state value stream
        self.value_stream = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Fully connected layers for advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(self, input_channels, action_space, model_file):
        self.global_step = 0
        self.state_dim = input_channels
        self.action_space = action_space
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.eval_net = DuelingDQN(input_channels, action_space).to(device)
        self.target_net = DuelingDQN(input_channels, action_space).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.epsilon = INITIAL_EPSILON
        self.model_file = model_file
        self.writer = SummaryWriter(log_dir='./logs', purge_step=self.global_step)

        # Load checkpoint or model
        self.load_checkpoint_or_model()

    def load_checkpoint_or_model(self):
        checkpoint_dir = os.path.join(self.model_file)
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if
                           f.startswith("checkpoint_step_") and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Loading checkpoint from {checkpoint_path}...\n")
                self.eval_net.load_state_dict(torch.load(checkpoint_path))

                step_path = os.path.join(self.model_file, "last_step.txt")
                if os.path.exists(step_path):
                    with open(step_path, "r") as f:
                        self.global_step = int(f.read().strip())
                else:
                    self.global_step = 0
            else:
                print("No checkpoints found. Trying to load existing model...\n")
                self.load_model()
        else:
            print("No checkpoint directory found. Trying to load existing model...\n")
            self.load_model()
            self.global_step = 0  # 新训练的步数从0开始

    def load_model(self):
        if os.path.exists(self.model_file) and not os.path.isdir(self.model_file):
            print("Loading model...\n")
            self.eval_net.load_state_dict(torch.load(self.model_file))
        else:
            print("No model file found. Starting from scratch.\n")

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state, action_mask):
        if random.random() <= self.epsilon:
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            action = random.choice(valid_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            if len(state.shape) == 4:
                q_values = self.eval_net(state).squeeze(0)
                action_mask_tensor = torch.FloatTensor(action_mask).to(device)
                masked_q_values = q_values * action_mask_tensor + (1 - action_mask_tensor) * (-1e9)
                action = torch.argmax(masked_q_values).item()
            else:
                raise ValueError("State input must have 4 dimensions: [batch, channels, height, width]")
        self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / 5000)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size, step):
        if len(self.replay_buffer) < batch_size:
            return
        minibatch = random.sample(self.replay_buffer, batch_size)
        state_batch = torch.FloatTensor(np.array([data[0] for data in minibatch])).to(device)
        action_batch = torch.LongTensor(np.array([data[1] for data in minibatch])).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array([data[2] for data in minibatch])).to(device)
        next_state_batch = torch.FloatTensor(np.array([data[3] for data in minibatch])).to(device)
        done_batch = torch.FloatTensor(np.array([data[4] for data in minibatch])).to(device)

        q_values = self.eval_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        loss = functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        total_norm = utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
        self.writer.add_scalar('Epsilon', self.epsilon, self.global_step)
        self.writer.add_scalar('Total reward', torch.sum(reward_batch).item(), self.global_step)
        self.writer.add_scalar('Max Q value', q_values.max().item(), self.global_step)
        self.writer.add_scalar('Gradient norm', total_norm, self.global_step)

        self.save_checkpoint(step)

        self.global_step += 1

    def save_checkpoint(self, step):
        checkpoint_path = f"./{self.model_file}/checkpoint_step_{step}.pth"
        torch.save(self.eval_net.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")

        with open(f"{self.model_file}/last_step.txt", "w") as f:
            f.write(str(step))

        checkpoint_dir = os.path.join(self.model_file)
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith('.pth')]
        if len(checkpoints) > 5:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            oldest_checkpoint = checkpoints[0]
            os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
            print(f"Deleted oldest checkpoint: {oldest_checkpoint}")

    def save_model(self, episode):
        filename = f"./{self.model_file}/dueling_dqn_trained_episode_{episode}.pth"
        torch.save(self.eval_net.state_dict(), filename)
        print("Model saved to", filename)
