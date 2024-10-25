import cv2
import time
import torch
import logging
import torch.nn.functional as F
import numpy as np
from input_keys import left_click, clear_action_state
from dueling_dqn_manual import keyboard_result, mouse_result
from dueling_dqn import DQNAgent, SMALL_BATCH_SIZE
from tool_manager import ToolManager
from game_control import take_action, pause_game, restart
from grab_screen import grab_full_screen, grab_region, extract_health, extract_posture, get_remaining_uses

logging.basicConfig(level=logging.INFO)


class GameEnvironment:
    def __init__(self, width=96, height=88, episodes=3000):
        self.width = width
        self.height = height
        self.episodes = episodes
        self.regions = {
            'self_blood': (55, 562, 399, 576),
            'boss_blood': (57, 91, 290, 106),
            'self_posture': (395, 535, 635, 552),
            'boss_posture': (315, 73, 710, 88),
            'remaining_uses': (955, 570, 971, 588)
        }
        self.paused = True
        self.manual = False
        self.debugged = False
        self.waiting_for_health_restore = False
        self.tool_manager = None
        self.self_stop_mark = 0
        self.target_step = 0
        self.heal_cooldown = 0
        self.heal_count = 9
        self.action_space_size = 8
        self.current_remaining_uses = 19

    def grab_screens(self):
        full_screen_img = grab_full_screen()
        screens = {key: grab_region(full_screen_img, region) for key, region in self.regions.items()}
        # Separate 'remaining_uses' from the screens used for the model
        remaining_uses_img = screens.pop('remaining_uses', None)
        return screens, remaining_uses_img

    @staticmethod
    def extract_features(screens):
        self_blood, boss_blood = extract_health(
            screens['self_blood'], screens['boss_blood'])
        self_posture, boss_posture = extract_posture(
            screens['self_posture'], screens['boss_posture'])
        return {'self_hp': self_blood, 'boss_hp': boss_blood,
                'self_posture': self_posture, 'boss_posture': boss_posture}

    def resize_screens(self, screens):
        resized_screens = {}
        for key, img in screens.items():
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            resized_img = F.interpolate(img_tensor, size=(self.width, self.height), mode='bilinear',
                                        align_corners=False)
            resized_screens[key] = resized_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return resized_screens

    @staticmethod
    def merge_states(screens):
        # Ensure 'remaining_uses' is not included in the model input
        return np.concatenate([img.transpose(2, 0, 1) for img in screens.values()], axis=0)

    def reset_marks(self):
        self.self_stop_mark = 0
        self.target_step = 0

    def get_action_mask(self):
        action_mask = [1] * self.action_space_size
        if self.tool_manager.tools_exhausted:
            action_mask[5] = action_mask[6] = action_mask[7] = 0
        else:
            remaining_cooldowns = self.tool_manager.get_remaining_cooldown()
            for i in range(3):
                if remaining_cooldowns[i] > 0:
                    action_mask[5 + i] = 0
        if self.heal_count == 0 or self.heal_cooldown > 0:
            action_mask[4] = 0
        return action_mask

    def set_tool_manager(self, tool_manager):
        self.tool_manager = tool_manager

    def update_remaining_uses(self, remaining_uses_img):
        self.current_remaining_uses = get_remaining_uses(remaining_uses_img, self.current_remaining_uses)
        if hasattr(self.tool_manager, 'remaining_uses'):
            self.tool_manager.remaining_uses = self.current_remaining_uses


class GameAgent:
    def __init__(self, input_channels=12, action_space=8, model_file="./models"):
        self.dqn_agent = DQNAgent(input_channels, action_space, model_file)
        self.TRAIN_BATCH_SIZE = SMALL_BATCH_SIZE

    def choose_action(self, state, action_mask):
        return self.dqn_agent.choose_action(state, action_mask)

    def store_transition(self, *args):
        self.dqn_agent.store_transition(*args)

    def train(self, episode):
        self.dqn_agent.train(self.TRAIN_BATCH_SIZE, episode)

    def update_target_network(self):
        self.dqn_agent.update_target_network()

    def save_model(self, episode):
        self.dqn_agent.save_model(episode)


class GameState:
    def __init__(self, features):
        self.current = features
        self.next = {}

    def update(self, features):
        self.next = features


class GameController:
    def __init__(self):
        self.total_reward = 0
        self.defeated = 0
        self.remaining_uses = 0
        self.env = GameEnvironment()
        self.tool_manager = ToolManager()
        self.env.set_tool_manager(self.tool_manager)
        self.agent = GameAgent()

    def action_judge(self, state):
        if self.env.waiting_for_health_restore:
            return 0, 0

        if state.next['self_hp'] < 1:
            reward, defeated = -12, 1
            self.total_reward += reward

        elif state.next['boss_hp'] < 1:
            reward, defeated = 50, 2
            self.total_reward += reward

        else:
            deltas = {key: state.next[key] - state.current[key] for key in state.current}
            reward = self.calculate_reward(deltas)
            defeated = 0
            self.total_reward += reward
            print(f'Current reward: {reward:.2f}, Total accumulated reward: {self.total_reward:.2f}')
        return reward, defeated

    def calculate_reward(self, deltas):
        reward = 0
        if deltas['self_hp'] <= -6 and self.env.self_stop_mark == 0:
            reward -= 4
            self.env.self_stop_mark = 1
        else:
            self.env.self_stop_mark = 0

        if deltas['boss_hp'] <= -2:
            reward += 12

        if deltas['self_posture'] > 9:
            reward -= 3

        if deltas['boss_posture'] > 5:
            reward += 9

        return reward

    def run(self):
        for episode in range(self.env.episodes):
            self.env.reset_marks()
            print("Press 'T' to start the screen capture")
            self.env.paused = pause_game(self.env.paused)

            last_time = time.time()

            # Initial screen capture and feature extraction
            screens, remaining_uses_img = self.env.grab_screens()
            features = self.env.extract_features(screens)
            state_obj = GameState(features)

            while True:
                self.env.target_step += 1
                cv2.waitKey(1)

                if self.env.waiting_for_health_restore:
                    if state_obj.current['self_hp'] > 30:
                        self.env.waiting_for_health_restore = False
                        print("Player health restored. Resuming normal operation.")
                        time.sleep(2)
                    else:
                        print("Waiting for player health to restore...")
                        screens, remaining_uses_img = self.env.grab_screens()
                        features = self.env.extract_features(screens)
                        state_obj = GameState(features)
                        left_click()
                        time.sleep(2)
                        continue

                # Process screen data and state
                resized_screens = self.env.resize_screens(screens)
                state = self.env.merge_states(resized_screens)

                if self.env.target_step % 10 == 0:
                    processing_time = time.time() - last_time
                    logging.info(f'Processing time: {processing_time:.2f}s in episode {episode}')
                    last_time = time.time()

                if self.env.heal_cooldown > 0:
                    self.env.heal_cooldown -= 1

                action_mask = self.env.get_action_mask()

                if self.env.manual:
                    action = self.get_manual_action()
                else:
                    action = self.agent.choose_action(state, action_mask)

                if not self.env.manual and action is not None:
                    take_action(action, self.env.debugged, self.tool_manager)

                # Capture screens and update 'remaining_uses'
                screens, remaining_uses_img = self.env.grab_screens()

                if self.env.target_step % 10 == 0:
                    self.env.update_remaining_uses(remaining_uses_img)

                # Get next state and update
                features = self.env.extract_features(screens)
                resized_screens = self.env.resize_screens(screens)
                next_state = self.env.merge_states(resized_screens)
                state_obj.update(features)

                # Evaluate reward and defeated status
                reward, self.defeated = self.action_judge(state_obj)

                if action == 5:
                    self.env.heal_count -= 1
                    self.env.heal_cooldown = 10
                    if state_obj.current['self_hp'] < 50:
                        reward += 6
                    else:
                        reward -= 10

                if action is not None:
                    self.agent.store_transition(state, action, reward, next_state, self.defeated)

                if self.env.target_step % self.agent.TRAIN_BATCH_SIZE == 0:
                    self.agent.train(episode)

                if self.defeated:
                    clear_action_state()
                    break

                # Pause control
                self.env.paused = pause_game(self.env.paused)
                state_obj.current = state_obj.next.copy()

            self.post_episode_updates(episode)
            restart(self.env.debugged, self.defeated)
            self.env.waiting_for_health_restore = True

        cv2.destroyAllWindows()

    @staticmethod
    def get_manual_action():
        if keyboard_result is not None:
            action = keyboard_result
            print(f'Keyboard action detected: {action}')
            return action
        elif mouse_result is not None:
            action = mouse_result
            print(f'Mouse action detected: {action}')
            return action

    def post_episode_updates(self, episode):
        if episode % 10 == 0 and not self.env.debugged:
            self.agent.update_target_network()
        if episode % 40 == 0 and not self.env.debugged:
            self.agent.save_model(episode)


if __name__ == '__main__':
    GameController().run()
