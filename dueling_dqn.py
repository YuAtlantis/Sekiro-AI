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
# tensorboard --logdir=logs
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

        # State value stream
        v = self.value_stream(x)

        # Advantage stream
        a = self.advantage_stream(x)

        # Combine streams to get Q value
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgent:
    def __init__(self, input_channels, action_space, model_file):
        self.state_dim = input_channels
        self.action_space = action_space

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)

        # Initialize networks
        self.eval_net = DuelingDQN(input_channels, action_space).to(device)
        self.target_net = DuelingDQN(input_channels, action_space).to(device)
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)

        # Epsilon for epsilon-greedy policy
        self.epsilon = INITIAL_EPSILON
        self.model_file = model_file
        self.writer = SummaryWriter(log_dir='./logs')

        if os.path.exists(self.model_file):
            if os.path.isdir(self.model_file):
                files = [f for f in os.listdir(self.model_file) if f.endswith('.pth')]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_file, x)), reverse=True)
                    model_path = os.path.join(self.model_file, files[0])
                    print(f"Loading model from {model_path}...\n")
                    self.eval_net.load_state_dict(torch.load(model_path))
                else:
                    print("No model files found in the folder.\n")
            else:
                print("Model exists, loading model...\n")
                self.eval_net.load_state_dict(torch.load(self.model_file))

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state):
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_space - 1)
        else:
            # Greedy action
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to device
            if len(state.shape) == 4:  # Ensure input is 4D (batch, channels, height, width)
                q_values = self.eval_net(state)
                action = torch.argmax(q_values, dim=1).item()
            else:
                raise ValueError("State input must have 4 dimensions: [batch, channels, height, width]")
        # Update epsilon
        self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / 1000)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size, step):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # Convert lists of numpy arrays to numpy arrays before converting to tensors
        state_batch = torch.FloatTensor(np.array([data[0] for data in minibatch])).to(device)
        action_batch = torch.LongTensor(np.array([data[1] for data in minibatch])).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array([data[2] for data in minibatch])).to(device)
        next_state_batch = torch.FloatTensor(np.array([data[3] for data in minibatch])).to(device)
        done_batch = torch.FloatTensor(np.array([data[4] for data in minibatch])).to(device)

        # Compute current Q values
        q_values = self.eval_net(state_batch).gather(1, action_batch).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        # Compute loss
        loss = functional.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Record the gradient norm (to track if gradients are too large or too small)
        total_norm = utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Write loss to TensorBoard
        self.writer.add_scalar('Loss/train', loss.item(), step)
        self.writer.add_scalar('Epsilon', self.epsilon, step)
        self.writer.add_scalar('Total reward', torch.sum(reward_batch).item(), step)
        self.writer.add_scalar('Max Q value', q_values.max().item(), step)
        self.writer.add_scalar('Gradient norm', total_norm, step)

    def save_model(self, episode):
        filename = f"./{self.model_file}/dueling_dqn_trained_episode_{episode}.pth"
        torch.save(self.eval_net.state_dict(), filename)
        print("Model saved to", filename)
