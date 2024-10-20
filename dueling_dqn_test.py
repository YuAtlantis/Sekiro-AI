import cv2
import time
import numpy as np
import dueling_dqn
from game_control import take_action, pause_game, restart
from grab_screen import grab, extract_self_and_boss_blood, extract_posture_bar


class GameEnvironment:
    def __init__(self, width=96, height=88, episodes=3000):
        self.width = width
        self.height = height
        self.episodes = episodes
        self.self_blood_window = (58, 562, 398, 575)
        self.boss_blood_window = (58, 92, 290, 108)
        self.self_posture_window = (395, 535, 635, 552)
        self.boss_posture_window = (315, 73, 710, 88)
        self.paused = True
        self.debugged = False
        self.self_stop_mark = 0
        self.boss_stop_mark = 0
        self.target_step = 0

    def grab_screen_state(self):
        return grab(self.self_blood_window), grab(self.boss_blood_window)

    def grab_screen_posture(self):
        return grab(self.self_posture_window), grab(self.boss_posture_window)

    def extract_health(self, self_blood_screen, boss_blood_screen):
        return extract_self_and_boss_blood(self_blood_screen, boss_blood_screen)

    def extract_posture(self, self_posture_screen, boss_posture_screen):
        return extract_posture_bar(self_posture_screen, boss_posture_screen)

    def resize_state(self, screen):
        return cv2.resize(screen, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    def grab_all_screens(self):
        self_blood_screen, boss_blood_screen = self.grab_screen_state()
        self_posture_screen, boss_posture_screen = self.grab_screen_posture()

        return self_blood_screen, boss_blood_screen, self_posture_screen, boss_posture_screen

    def resize_all_screens(self, self_blood_screen, boss_blood_screen, self_posture_screen, boss_posture_screen):
        self_blood_screen_resized = self.resize_state(self_blood_screen)
        boss_blood_screen_resized = self.resize_state(boss_blood_screen)
        self_posture_screen_resized = self.resize_state(self_posture_screen)
        boss_posture_screen_resized = self.resize_state(boss_posture_screen)

        return self_blood_screen_resized, boss_blood_screen_resized, self_posture_screen_resized, boss_posture_screen_resized

    def merge_all_states(self, self_blood, boss_blood, self_posture, boss_posture):
        self_blood = self_blood.transpose(2, 0, 1)
        boss_blood = boss_blood.transpose(2, 0, 1)
        self_posture = self_posture.transpose(2, 0, 1)
        boss_posture = boss_posture.transpose(2, 0, 1)

        combined_state = np.concatenate((self_blood, boss_blood, self_posture, boss_posture), axis=0)  # 在通道维度上拼接
        return combined_state  # 返回形状应为 (channels, height, width)

    def reset_marks(self):
        self.self_stop_mark = 0
        self.boss_stop_mark = 0
        self.target_step = 0


class GameAgent:
    def __init__(self, input_channels=12, action_space=3, model_file="./models"):
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


class GameState:
    def __init__(self, self_hp, boss_hp, self_posture, boss_posture):
        self.self_hp = self_hp
        self.boss_hp = boss_hp
        self.self_posture = self_posture
        self.boss_posture = boss_posture
        self.next_self_hp = 0
        self.next_boss_hp = 0
        self.next_self_posture = 0
        self.next_boss_posture = 0

    def update(self, next_self_hp, next_boss_hp, next_self_posture, next_boss_posture):
        self.next_self_hp = next_self_hp
        self.next_boss_hp = next_boss_hp
        self.next_self_posture = next_self_posture
        self.next_boss_posture = next_boss_posture


class GameController:
    def __init__(self):
        self.env = GameEnvironment()
        self.agent = GameAgent()

    def action_judge(self, state: GameState):
        if state.next_self_hp < 2:  # Player dies
            reward = -10
            finish_flag = 1
            self.env.reset_marks()
            print(f'You are dead and get the reward: {reward:.2f}')
            return reward, finish_flag

        if state.next_boss_hp < 4:  # Defeat the boss
            reward = 20
            finish_flag = 1
            self.env.reset_marks()
            print(f'You finally beat the boss and get the reward: {reward:.2f}')
            return reward, finish_flag

        # Calculate health and posture changes
        delta_self = state.next_self_hp - state.self_hp
        delta_boss = state.next_boss_hp - state.boss_hp
        delta_self_posture = state.next_self_posture - state.self_posture
        delta_boss_posture = state.next_boss_posture - state.boss_posture

        reward = self.calculate_reward(delta_self, delta_boss, delta_self_posture, delta_boss_posture)
        finish_flag = 0
        print(f'Current reward is: {reward:.2f}')

        return reward, finish_flag

    def calculate_reward(self, delta_self, delta_boss, delta_self_posture, delta_boss_posture):
        reward = 0

        if delta_self <= -6:
            if self.env.self_stop_mark == 0:
                reward += -6
                self.env.self_stop_mark = 1
        else:
            self.env.self_stop_mark = 0

        if delta_boss <= -3:
            if self.env.boss_stop_mark == 0:
                reward += 8
                self.env.boss_stop_mark = 1
        else:
            self.env.boss_stop_mark = 0

        if delta_self_posture > 6:
            reward += -3
        if delta_boss_posture > 4:
            reward += 5

        return reward

    def run(self):
        for episode in range(self.env.episodes):
            self.env.reset_marks()
            print("You can press t to start the screen capture")
            self.env.paused = pause_game(self.env.paused)

            self_blood_screen, boss_blood_screen, self_posture_screen, boss_posture_screen = self.env.grab_all_screens()

            self_blood, boss_blood = self.env.extract_health(self_blood_screen, boss_blood_screen)
            self_posture, boss_posture = self.env.extract_posture(self_posture_screen, boss_posture_screen)
            last_time = time.time()

            game_state = GameState(self_blood, boss_blood, self_posture, boss_posture)

            while True:
                (self_blood_screen_resized, boss_blood_screen_resized, self_posture_screen_resized,
                 boss_posture_screen_resized) = self.env.resize_all_screens(self_blood_screen, boss_blood_screen,
                                                                            self_posture_screen, boss_posture_screen)

                state = self.env.merge_all_states(self_blood_screen_resized, boss_blood_screen_resized,
                                                  self_posture_screen_resized, boss_posture_screen_resized)

                print(f'Screen -> State takes {time.time() - last_time} seconds in episode {episode}')
                last_time = time.time()
                self.env.target_step += 1

                action = self.agent.choose_action(state)
                take_action(action, self.env.debugged)

                self_blood_screen, boss_blood_screen, self_posture_screen, boss_posture_screen = self.env.grab_all_screens()
                cv2.waitKey(1)

                (self_blood_screen_resized, boss_blood_screen_resized, self_posture_screen_resized,
                 boss_posture_screen_resized) = self.env.resize_all_screens(self_blood_screen, boss_blood_screen,
                                                                            self_posture_screen, boss_posture_screen)

                next_state = self.env.merge_all_states(self_blood_screen_resized, boss_blood_screen_resized,
                                                       self_posture_screen_resized, boss_posture_screen_resized)

                next_self_blood, next_boss_blood = self.env.extract_health(self_blood_screen, boss_blood_screen)
                next_self_posture, next_boss_posture = self.env.extract_posture(self_posture_screen,
                                                                                boss_posture_screen)

                game_state.update(next_self_blood, next_boss_blood, next_self_posture, next_boss_posture)

                action_reward, done = self.action_judge(game_state)
                self.agent.store_transition(state, action, action_reward, next_state, done)

                if self.env.target_step % self.agent.TRAIN_BATCH_SIZE == 0:
                    self.agent.train(episode)

                if done:
                    break

                self.env.paused = pause_game(self.env.paused)

            if episode % 10 == 0 and not self.env.debugged:
                self.agent.update_target_network()
            if episode % 50 == 0 and not self.env.debugged:
                self.agent.save_model(episode)

            restart(self.env.debugged)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    controller = GameController()
    controller.run()
