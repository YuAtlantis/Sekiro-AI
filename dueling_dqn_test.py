import cv2
import time
import numpy as np
import dueling_dqn
from game_control import take_action, pause_game, restart
from grab_screen import grab, extract_self_and_boss_blood


class GameEnvironment:
    def __init__(self, width=96, height=88, episodes=3000):
        self.width = width
        self.height = height
        self.episodes = episodes
        self.self_blood_window = (100, 650, 450, 665)
        self.boss_blood_window = (100, 180, 340, 195)
        self.paused = True
        self.debugged = False
        self.self_stop_mark = 0
        self.boss_stop_mark = 0
        self.target_step = 0

    def grab_screen_state(self):
        self_screen = grab(self.self_blood_window)
        boss_screen = grab(self.boss_blood_window)
        return self_screen, boss_screen

    def extract_health(self, self_screen, boss_screen):
        return extract_self_and_boss_blood(self_screen, boss_screen)

    def resize_state(self, screen):
        return cv2.resize(screen, (self.width, self.height))

    def reset_marks(self):
        self.self_stop_mark = 0
        self.boss_stop_mark = 0
        self.target_step = 0


class GameAgent:
    def __init__(self, input_channels=3, action_space=4, model_file="./models"):
        self.agent = dueling_dqn.DQNAgent(input_channels, action_space, model_file)
        self.TRAIN_BATCH_SIZE = dueling_dqn.SMALL_BATCH_SIZE

    def choose_action(self, state):
        return self.agent.choose_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.agent.store_transition(state, action, reward, next_state, done)

    def train(self, episode):
        self.agent.train(self.TRAIN_BATCH_SIZE, episode)

    def update_target_network(self):
        self.agent.update_target_network()

    def save_model(self, episode):
        self.agent.save_model(episode)


class GameController:
    def __init__(self):
        self.env = GameEnvironment()
        self.agent = GameAgent()

    def action_judge(self, boss_hp, next_boss_hp, self_hp, next_self_hp):
        # Player dies
        if next_self_hp < 2:
            reward = -10
            finish_flag = 1
            self.env.self_stop_mark = 0
            self.env.boss_stop_mark = 0
            print(f'You are dead and get the reward: {reward:.2f}')
            return reward, finish_flag

        # Defeat the boss
        if next_boss_hp < 4:
            reward = 20
            finish_flag = 1
            self.env.self_stop_mark = 0
            self.env.boss_stop_mark = 0
            print(f'You finally beat the boss and get the reward: {reward:.2f}')
            return reward, finish_flag

        # Calculate health changes
        delta_self = next_self_hp - self_hp
        delta_boss = next_boss_hp - boss_hp

        print(f'delta_self: {delta_self}')
        print(f'delta_boss: {delta_boss}')

        # Initialize reward
        reward = 0

        # Player's health decreases
        if delta_self <= -6:
            if self.env.self_stop_mark == 0:
                reward += -6
                self.env.self_stop_mark = 1
        else:
            self.env.self_stop_mark = 0

        # Boss's health decreases
        if delta_boss <= -3:
            if self.env.boss_stop_mark == 0:
                reward += 8
                self.env.boss_stop_mark = 1
        else:
            self.env.boss_stop_mark = 0

        finish_flag = 0
        print(f'Current reward is: {reward:.2f}')

        return reward, finish_flag

    def run(self):
        for episode in range(self.env.episodes):
            self.env.reset_marks()

            print("Press t to start the screen capture")
            self.env.paused = pause_game(self.env.paused)
            self_screen, boss_screen = self.env.grab_screen_state()
            self_blood, boss_blood = self.env.extract_health(self_screen, boss_screen)

            state = self.env.resize_state(self_screen)
            last_time = time.time()

            while True:
                # Limit the frequency of screen capture
                # if time.time() - last_time < 0.5:
                #     continue

                state = np.array(state).reshape(3, self.env.height, self.env.width)
                print('Screen -> State takes {} seconds in episode {}'.format(time.time() - last_time, episode))
                last_time = time.time()
                self.env.target_step += 1

                # Choose an action based on the current state
                action = self.agent.choose_action(state)

                # Execute action
                take_action(action, self.env.debugged)

                self_screen, boss_screen = self.env.grab_screen_state()
                cv2.waitKey(1)

                next_state = self.env.resize_state(self_screen)
                next_state = np.array(next_state).reshape(3, self.env.height, self.env.width)

                next_self_blood, next_boss_blood = self.env.extract_health(self_screen, boss_screen)

                action_reward, done = self.action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood)
                self.agent.store_transition(state, action, action_reward, next_state, done)

                if self.env.target_step % self.agent.TRAIN_BATCH_SIZE == 0:
                    self.agent.train(episode)

                # Update self_blood and boss_blood
                self_blood = next_self_blood
                boss_blood = next_boss_blood

                if done == 1:
                    break

                self.env.paused = pause_game(self.env.paused)

            if episode != 0 and not self.env.debugged:
                if episode % 10 == 0:
                    self.agent.update_target_network()
                if episode % 50 == 0:
                    self.agent.save_model(episode)

            restart(self.env.debugged)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    controller = GameController()
    controller.run()
