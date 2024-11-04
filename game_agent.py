# game_agent.py

import logging
from dqn.dueling_dqn import DQNAgent, BIG_BATCH_SIZE

logging.basicConfig(level=logging.INFO)


class GameAgent:
    def __init__(self, input_channels=3, action_space=7, model_file="./models",
                 model_folder="./models"):
        self.dqn_agent = DQNAgent(input_channels, action_space, model_file, model_folder)
        self.TRAIN_BATCH_SIZE = BIG_BATCH_SIZE

    def choose_action(self, state, action_mask):
        """Choose an action based on the current state and action mask."""
        return self.dqn_agent.choose_action(state, action_mask)

    def store_transition(self, *args):
        """Store a transition in the replay buffer."""
        self.dqn_agent.store_transition(*args)

    def train(self):
        """Train the DQN agent."""
        self.dqn_agent.train(self.TRAIN_BATCH_SIZE)

    def update_target_network(self):
        """Update the target network."""
        self.dqn_agent.update_target_network()

    def save_model(self, episode):
        """Save the model."""
        self.dqn_agent.save_model(episode)
