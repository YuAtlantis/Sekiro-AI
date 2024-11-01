import cv2
import time
import torch
import logging
import torch.nn.functional as F
from input_keys import clear_action_state, attack
from dueling_dqn_manual import keyboard_result, mouse_result, start_listeners
from dueling_dqn import DQNAgent, BIG_BATCH_SIZE
from tool_manager import ToolManager
from game_control import take_action, pause_game, restart
from grab_screen import grab_full_screen, grab_region, extract_health, extract_posture, get_remaining_uses

logging.basicConfig(level=logging.INFO)


class GameEnvironment:
    def __init__(self, width=128, height=128, episodes=3000):
        self.width = width
        self.height = height
        self.episodes = episodes
        self.regions = {
            'game_window': (220, 145, 800, 530),
            'self_blood': (55, 562, 399, 576),
            'boss_blood': (57, 92, 290, 106),
            'self_posture': (395, 535, 635, 552),
            'boss_posture': (315, 73, 710, 88),
            'remaining_uses': (955, 570, 971, 588)
        }
        self.paused = True
        self.manual = False
        self.debugged = False
        self.tool_manager = None
        self.single_life_boss = True
        self.self_stop_mark = 0
        self.target_step = 0
        self.heal_cooldown = 5
        self.heal_count = 9999
        self.action_space_size = 8
        self.current_remaining_uses = 19

    def grab_screens(self):
        full_screen_img = grab_full_screen()
        screens = {key: grab_region(full_screen_img, region) for key, region in self.regions.items()}
        remaining_uses_img = screens.pop('remaining_uses', None)
        game_window_img = screens.pop('game_window')
        return game_window_img, screens, remaining_uses_img

    @staticmethod
    def extract_features(screens):
        self_blood, boss_blood = extract_health(
            screens['self_blood'], screens['boss_blood'])
        self_posture, boss_posture = extract_posture(
            screens['self_posture'], screens['boss_posture'])
        return {'self_hp': self_blood, 'boss_hp': boss_blood,
                'self_posture': self_posture, 'boss_posture': boss_posture}

    def resize_screen(self, img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        resized_img = F.interpolate(img_tensor, size=(self.height, self.width), mode='bilinear',
                                    align_corners=False)
        resized_img = resized_img.squeeze(0).cpu().numpy()
        return resized_img

    @staticmethod
    def prepare_state(img):
        img_tensor = torch.from_numpy(img).float() / 255.0  # Already in [C, H, W] format for RGB images

        # Ensure img_tensor has a batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

        num_channels = img_tensor.shape[1]  # Should be 3 for RGB images

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)

        # Adjust mean and std if num_channels is not 3
        if num_channels != 3:
            mean = mean[:, :num_channels, :, :]
            std = std[:, :num_channels, :, :]

        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.squeeze(0)  # [C, H, W]

        return img_tensor

    def reset_marks(self):
        self.self_stop_mark = 0
        self.target_step = 0
        self.heal_count = 9999

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
    def __init__(self, input_channels=3, action_space=8, model_file="./models/dueling_dqn_trained_episode_120.pth",
                 model_folder="./models"):
        self.dqn_agent = DQNAgent(input_channels, action_space, model_file, model_folder)
        self.TRAIN_BATCH_SIZE = BIG_BATCH_SIZE

    def choose_action(self, state, action_mask):
        return self.dqn_agent.choose_action(state, action_mask)

    def store_transition(self, *args):
        self.dqn_agent.store_transition(*args)

    def train(self):
        self.dqn_agent.train(self.TRAIN_BATCH_SIZE)

    def update_target_network(self):
        self.dqn_agent.update_target_network()

    def save_model(self, episode):
        self.dqn_agent.save_model(episode)


class GameState:
    def __init__(self, features, state):
        self.current_features = features
        self.next_features = features.copy()
        self.current_state = state
        self.next_state = state.clone()

    def update(self, features, state):
        self.current_features = self.next_features.copy()
        self.next_features = features.copy()
        self.current_state = self.next_state.clone()
        self.next_state = state.clone()


class GameController:
    def __init__(self):
        self.total_reward = 0
        self.defeated = 0
        self.defeat_count = 0
        self.missing_boss_hp_steps = 0
        self.defeat_window_start = None
        self.phase_transition_detected = False
        self.env = GameEnvironment()
        self.tool_manager = ToolManager()
        self.env.set_tool_manager(self.tool_manager)
        self.agent = GameAgent()
        self.reward_weights = {
            'self_hp_loss': -1.0,
            'boss_hp_loss': 1.0,
            'self_death': -20,
            'self_posture_increase': -0.6,
            'boss_posture_increase': 0.6,
            'defeat_bonus': 50,
            'survival_reward': 0.1,
            'successful_defense': 1.0,
        }

        self.reward_type_distribution = {
            'self_hp_loss': [],
            'boss_hp_loss': [],
            'self_posture_increase': [],
            'boss_posture_increase': [],
            'defeat_bonus': [],
            'self_death': [],
        }
        self.current_reward_types = {key: 0 for key in self.reward_weights}

    def action_judge(self, state_obj):
        reward, defeated = 0, 0
        self.defeat_window_start = None
        self.phase_transition_detected = False

        survival_reward = self.reward_weights['survival_reward']
        reward += survival_reward
        self.current_reward_types['survival_reward'] += survival_reward

        if self.check_successful_defense(state_obj):
            defense_reward = self.reward_weights['successful_defense']
            reward += defense_reward
            self.current_reward_types['successful_defense'] += defense_reward
        else:
            defense_reward = 0

        if state_obj.next_features['self_hp'] <= 0.1:
            death_penalty = self.reward_weights['self_death']
            reward += death_penalty
            self.current_reward_types['self_death'] += death_penalty
            defeated = 1
        elif state_obj.next_features['boss_hp'] <= 0.1:
            defeat_reward, defeated = self.handle_boss_low_health(state_obj)
            reward += defeat_reward
        else:
            delta_reward = self.calculate_deltas(state_obj)
            reward += delta_reward

        self.total_reward += reward
        return reward, defeated

    @staticmethod
    def check_successful_defense(state_obj):
        hp_delta = state_obj.next_features['self_hp'] - state_obj.current_features['self_hp']
        posture_delta = state_obj.next_features['self_posture'] - state_obj.current_features['self_posture']

        if hp_delta < 1 and posture_delta > 0:
            return True
        else:
            return False

    def handle_boss_low_health(self, state_obj):
        if self.defeat_count >= 2 or self.env.single_life_boss:
            reward = self.reward_weights['defeat_bonus']
            self.current_reward_types['defeat_bonus'] += reward
            self.defeated = 2
            print("Boss defeated, game over")
            self.defeat_window_start = None
        else:
            if not self.defeat_window_start:
                self.defeat_window_start = time.time()
            print("Boss HP <= 0, attempting to kill")
            reward = self.attack_in_low_health_phase(state_obj)
            self.defeated = 0
        return reward, self.defeated

    def attack_in_low_health_phase(self, state_obj):
        reward = 0
        time_elapsed = time.time() - self.defeat_window_start if self.defeat_window_start else 0

        if self.env.single_life_boss:
            self.missing_boss_hp_steps += 1
        elif state_obj.next_features['boss_hp'] <= 0.1:
            self.missing_boss_hp_steps += 1
        else:
            self.missing_boss_hp_steps = 0

        if self.missing_boss_hp_steps > 256:
            print("Boss血条已消失，停止游戏")
            self.defeated = 2
            return reward

        if not self.env.single_life_boss and state_obj.current_features['boss_hp'] > 50:
            if not self.phase_transition_detected:
                reward = self.reward_weights['defeat_bonus']
                self.current_reward_types['defeat_bonus'] += reward
                self.defeat_count += 1
                self.phase_transition_detected = True
                print(f"Boss HP restored above 50%, entering next phase: {self.defeat_count}")
                self.defeat_window_start = None
        else:
            if time_elapsed < 8:
                attack()
                print("Continuing to attack Boss, ensuring defeat...")
            else:
                self.defeat_window_start = None
                self.phase_transition_detected = False
        return reward

    def calculate_deltas(self, state_obj):
        keys = ['self_hp', 'boss_hp', 'self_posture', 'boss_posture']
        deltas = {key: state_obj.next_features[key] - state_obj.current_features[key] for key in keys}
        reward = 0

        if deltas['self_hp'] < 0:
            self_hp_loss = self.reward_weights['self_hp_loss'] * abs(deltas['self_hp'])
            reward += self_hp_loss
            self.current_reward_types['self_hp_loss'] += self_hp_loss
        else:
            self_hp_loss = 0

        if deltas['boss_hp'] < 0:
            boss_hp_reward = self.reward_weights['boss_hp_loss'] * abs(deltas['boss_hp'])
            reward += boss_hp_reward
            self.current_reward_types['boss_hp_loss'] += boss_hp_reward
        else:
            boss_hp_reward = 0

        if deltas['self_posture'] > 0 and state_obj.current_features['self_posture'] > 80:
            self_posture_penalty = self.reward_weights['self_posture_increase'] * deltas['self_posture']
            reward += self_posture_penalty
            self.current_reward_types['self_posture_increase'] += self_posture_penalty
        else:
            self_posture_penalty = 0

        if deltas['boss_posture'] > 0:
            boss_posture_reward = self.reward_weights['boss_posture_increase'] * deltas['boss_posture']
            reward += boss_posture_reward
            self.current_reward_types['boss_posture_increase'] += boss_posture_reward
        else:
            boss_posture_reward = 0

        logging.info(
            f"步骤 {self.env.target_step}: 奖励细节 - "
            f"自身体力损失惩罚: {self_hp_loss:.2f}, "
            f"Boss体力损失奖励: {boss_hp_reward:.2f}, "
            f"自身姿势增加惩罚: {self_posture_penalty:.2f}, "
            f"Boss姿势增加奖励: {boss_posture_reward:.2f}"
        )

        return reward

    def post_episode_updates(self, episode):
        for key in self.reward_type_distribution:
            self.reward_type_distribution[key].append(self.current_reward_types.get(key, 0))

        reward_details = " | ".join([f"{key}: {value:.2f}" for key, value in self.current_reward_types.items()])
        reward_summary = f"第 {episode + 1} 回合总结: 总奖励: {self.total_reward:.2f} | {reward_details}"

        print(reward_summary)
        logging.info(reward_summary)

        self.total_reward = 0
        self.current_reward_types = {key: 0 for key in self.reward_weights}

        if episode % 10 == 0 and not self.env.debugged:
            self.agent.update_target_network()
        if episode % 50 == 0 and not self.env.debugged:
            self.agent.save_model(episode)

    def run(self):
        if self.env.manual:
            start_listeners()
        print("Press 'T' to start the screen capture")
        for episode in range(self.env.episodes):
            self.env.reset_marks()
            self.env.paused = pause_game(self.env.paused)
            clear_action_state()
            last_time = time.time()

            game_window_img, screens, remaining_uses_img = self.env.grab_screens()
            features = self.env.extract_features(screens)
            resized_img = self.env.resize_screen(game_window_img)
            state = self.env.prepare_state(resized_img)
            state_obj = GameState(features, state)

            while True:
                self.env.target_step += 1
                self.env.paused = pause_game(self.env.paused)
                cv2.waitKey(1)

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
                    action = self.agent.choose_action(state_obj.current_state, action_mask)

                if not self.env.manual and action is not None:
                    take_action(action, self.env.debugged, self.tool_manager)

                game_window_img, screens, remaining_uses_img = self.env.grab_screens()

                if self.env.target_step % 10 == 0:
                    self.env.update_remaining_uses(remaining_uses_img)

                features = self.env.extract_features(screens)
                resized_img = self.env.resize_screen(game_window_img)
                next_state = self.env.prepare_state(resized_img)
                state_obj.update(features, next_state)

                reward, self.defeated = self.action_judge(state_obj)

                if action == 4:
                    self.env.heal_count -= 1
                    self.env.heal_cooldown = 15
                    if state_obj.current_features['self_hp'] < 50:
                        reward += 2
                    else:
                        reward -= 5

                if action is not None:
                    self.agent.store_transition(state_obj.current_state, action, reward, state_obj.next_state,
                                                self.defeated)

                if self.env.target_step % 64 == 0:
                    self.agent.train()

                if self.defeated:
                    break

            clear_action_state()
            self.post_episode_updates(episode)
            restart(self.env, self.defeated, self.defeat_count)

        cv2.destroyAllWindows()

    @staticmethod
    def get_manual_action():
        if keyboard_result is not None:
            return keyboard_result
        elif mouse_result is not None:
            return mouse_result


if __name__ == '__main__':
    GameController().run()
