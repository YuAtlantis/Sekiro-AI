# game_agent.py

import logging
from dqn.dueling_dqn import DQNAgent, BIG_BATCH_SIZE


logging.basicConfig(level=logging.INFO)


class GameAgent:
    def __init__(self, input_channels=3, action_space=4, model_file="./models",
                 model_folder="./models"):
        self.dqn_agent = DQNAgent(input_channels, action_space, model_file, model_folder)
        self.TRAIN_BATCH_SIZE = BIG_BATCH_SIZE

    @property
    def global_episode(self):
        return self.dqn_agent.global_episode

    @global_episode.setter
    def global_episode(self, value):
        self.dqn_agent.global_episode = value

    def choose_action(self, state, action_mask):
        """Choose an action based on the current state and action mask."""
        return self.dqn_agent.choose_action(state, action_mask)

    def store_transition(self, *args):
        """Store a transition in the replay buffer."""
        self.dqn_agent.store_transition(*args)

    def train(self, batch_size=None, buffer_size=6000):
        """Train the DQN agent."""
        if batch_size is None:
            batch_size = self.TRAIN_BATCH_SIZE
        self.dqn_agent.train(batch_size, buffer_size)

    def update_target_network(self):
        """Update the target network."""
        self.dqn_agent.update_target_network()

    def log_episode_reward(self, episode, total_reward, moving_average):
        """Log the total reward and moving average reward to SummaryWriter."""
        self.dqn_agent.writer.add_scalar('Episode/TotalReward', total_reward, episode)
        self.dqn_agent.writer.add_scalar('Episode/MovingAverageReward', moving_average, episode)

    def close_writer(self):
        """Close the SummaryWriter."""
        self.dqn_agent.writer.close()
        logging.info("SummaryWriter closed.")
