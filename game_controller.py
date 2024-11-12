# game_controller.py

import cv2
import logging
import time
import numpy as np
from game_environment import GameEnvironment
from game_agent import GameAgent
from game_state import GameState
from control.tool_manager import ToolManager
from keys.input_keys import attack
from control.dueling_dqn_manual import keyboard_result, mouse_result, start_listeners
from control.game_control import take_action, pause_game, restart
from logging.handlers import RotatingFileHandler
from collections import deque

# Configure logging with rotating file handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('./logs/game_controller.log', maxBytes=10 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GameController:
    def __init__(self):
        self.last_feature_log_time = 0
        self.last_time_penalty_update = time.time()
        self.time_penalty_increment = -0.001

        self.last_actions = deque(maxlen=8)

        self.defeated = 0

        self.missing_boss_hp_steps = 0
        self.boss_lives = 1

        self.steps_since_last_attack = 0
        self.idle_threshold = 6

        self.defeat_window_start = None
        self.env = GameEnvironment()
        self.tool_manager = ToolManager()
        self.env.set_tool_manager(self.tool_manager)
        self.agent = GameAgent()
        self.intermediate_rewards_given = {
            '75%': False,
            '50%': False,
            '25%': False
        }
        self.reward_weights = {
            'self_hp_loss': -0.5,
            'boss_hp_loss': 10.0,
            'self_death': -12,
            'defeat_bonus': 35,
            'time_penalty': -0.001,
            "intermediate_defeat": 0,
            'idle_penalty': -3
        }

        self.reward_type_distribution = {
            'self_hp_loss': [],
            'boss_hp_loss': [],
            'defeat_bonus': [],
            'self_death': [],
            'idle_penalty': [],
            'time_penalty': [],
        }
        self.flags = {
            'self_hp_loss': False,
            'boss_hp_loss': False,
        }
        self.current_reward_types = {key: 0 for key in self.reward_weights}
        self.episode_rewards = deque(maxlen=100)
        self.moving_average_rewards = deque(maxlen=100)

    def action_judge(self, state_obj):
        """Judge the action and calculate the reward."""
        reward, defeated = 0, 0
        self.defeat_window_start = None

        # 1. Check for Boss Defeat
        if state_obj.next_features['boss_hp'] <= 0:
            defeat_reward, defeated = self.handle_boss_low_health(state_obj)
            reward += defeat_reward
        else:
            # 2. Calculate Deltas and Corresponding Rewards
            delta_reward = self.calculate_deltas(state_obj)
            reward += delta_reward

        if not self.env.manual:
            current_time = time.time()
            if current_time - self.last_time_penalty_update >= 1:
                # 3. Apply Time Penalty
                time_penalty = self.reward_weights['time_penalty'] + (self.env.target_step * self.time_penalty_increment)
                reward += time_penalty
                self.current_reward_types['time_penalty'] += time_penalty
                self.last_time_penalty_update = current_time

        # 4. Apply Idle Penalty
        if not self.env.manual:
            if self.steps_since_last_attack >= self.idle_threshold:
                idle_penalty = self.reward_weights['idle_penalty']
                reward += idle_penalty
                self.current_reward_types['idle_penalty'] += idle_penalty
                logger.info("Idle penalty applied due to prolonged inactivity.")
                self.steps_since_last_attack = 0

        # 5. Check for Death
        if state_obj.next_features['self_hp'] <= 1:
            death_penalty = self.reward_weights['self_death']
            reward += death_penalty
            self.current_reward_types['self_death'] += death_penalty
            defeated = 1
            logger.info("Agent has died. Death penalty applied.")
            return reward, defeated

        return reward, defeated

    def handle_boss_low_health(self, state_obj):
        boss_hp = state_obj.next_features.get('boss_hp', 0)
        defeat_bonus = self.reward_weights.get('defeat_bonus', 40)

        if self.boss_lives <= 1:
            logger.info("Boss is in the final phase and try to defeat directly!")
            reward = self.attack_directly()
            self.defeated = 2
            return reward, self.defeated
        else:
            if not self.defeat_window_start:
                self.defeat_window_start = time.time()
                logger.info(f"Boss HP ≤ 1, starting phase: {self.boss_lives}")

            if boss_hp <= 0:
                self.missing_boss_hp_steps += 1
            else:
                self.missing_boss_hp_steps = 0

            if self.missing_boss_hp_steps > 50:
                reward = defeat_bonus
                self.boss_lives -= 1
                self.defeated = 0
                logger.info("Boss defeated in this phase.")
                self.missing_boss_hp_steps = 0
                return reward, self.defeated
            else:
                reward, defeated = self.attack_in_low_health_phase(state_obj)
                self.defeated = defeated
                return reward, self.defeated

    def attack_directly(self):
        """Attack the boss directly to defeat it."""
        attack()
        defeat_bonus = self.reward_weights.get('defeat_bonus')
        reward = defeat_bonus
        self.current_reward_types['defeat_bonus'] += defeat_bonus
        logger.info("Boss defeated directly; defeat bonus awarded.")
        return reward

    def attack_in_low_health_phase(self, state_obj):
        """Attack during the boss's low health phase and send rewards when Boss enters the next phase."""
        boss_hp = state_obj.next_features.get('boss_hp', 0)
        defeat_bonus = self.reward_weights.get('defeat_bonus', 20)
        reward = 0
        time_elapsed = time.time() - self.defeat_window_start if self.defeat_window_start else 0

        if time_elapsed > 6:
            self.defeat_window_start = None
            logger.info("Defeat window expired, stopping attack.")
            return reward, self.defeated

        attack()
        logger.info("Continuing to attack Boss to ensure defeat...")

        if boss_hp > 50:
            logger.info(f"Boss entered the next phase, remaining lives: {self.boss_lives - 1}")
            reward = defeat_bonus
            self.boss_lives -= 1
            self.defeated = 0
            self.defeat_window_start = None
            self.intermediate_rewards_given = {
                '75%': False,
                '50%': False,
                '25%': False
            }
            self.missing_boss_hp_steps = 0
            return reward, self.defeated

        return reward, self.defeated

    def calculate_deltas(self, state_obj):
        """Calculate the reward based on the changes in features."""
        keys = ['self_hp', 'boss_hp']
        deltas = {key: state_obj.next_features[key] - state_obj.current_features[key] for key in keys}
        reward = 0

        # 1. Self HP loss penalty
        if -50 < deltas['self_hp'] < -10:
            if not self.flags['self_hp_loss']:
                self_hp_loss = self.reward_weights['self_hp_loss'] * abs(deltas['self_hp'])
                reward += self_hp_loss
                self.current_reward_types['self_hp_loss'] += self_hp_loss
                self.flags['self_hp_loss'] = True
                logger.info(f"Self HP reduced: {abs(deltas['self_hp']):.2f} ; penalty applied: {self_hp_loss:.2f}")
        else:
            self.flags['self_hp_loss'] = False

        # 2. Boss HP loss reward
        if -12 < deltas['boss_hp'] < -1:
            if not self.flags['boss_hp_loss']:
                boss_hp_reward = self.reward_weights['boss_hp_loss'] * abs(deltas['boss_hp'])
                reward += boss_hp_reward
                self.current_reward_types['boss_hp_loss'] += boss_hp_reward
                self.steps_since_last_attack = 0
                self.flags['boss_hp_loss'] = True
                logger.info(f"Boss HP reduced: {abs(deltas['boss_hp']):.2f} ; reward applied: {boss_hp_reward:.2f}")
        else:
            self.flags['boss_hp_loss'] = False

        # 3. Intermediate rewards based on boss HP thresholds
        boss_hp_percentage = state_obj.next_features['boss_hp']
        if 75 > boss_hp_percentage >= 50 and not self.intermediate_rewards_given['75%']:
            reward += 15
            self.current_reward_types['intermediate_defeat'] += 10
            self.intermediate_rewards_given['75%'] = True
            logger.info("Intermediate reward granted for boss HP between 50% and 75%.")
        elif 50 > boss_hp_percentage >= 25 and not self.intermediate_rewards_given['50%']:
            reward += 25
            self.current_reward_types['intermediate_defeat'] += 20
            self.intermediate_rewards_given['50%'] = True
            logger.info("Intermediate reward granted for boss HP between 25% and 50%.")
        elif boss_hp_percentage < 25 and not self.intermediate_rewards_given['25%']:
            reward += 35
            self.current_reward_types['intermediate_defeat'] += 30
            self.intermediate_rewards_given['25%'] = True
            logger.info("Intermediate reward granted for boss HP below 25%.")

        return reward

    def post_episode_updates(self, episode):
        """Update statistics and save models after each episode."""
        for key in self.reward_type_distribution:
            self.reward_type_distribution[key].append(self.current_reward_types.get(key, 0))

        total_reward = sum(self.current_reward_types.values())
        self.episode_rewards.append(total_reward)

        moving_average = np.mean(self.episode_rewards) if self.episode_rewards else 0
        self.moving_average_rewards.append(moving_average)

        reward_details = " | ".join([f"{key}: {value:.2f}" for key, value in self.current_reward_types.items()])
        reward_summary = (f"Episode {episode + 1} Summary: Total Reward: {total_reward:.2f} | "
                          f"Moving Average Reward (Last {len(self.episode_rewards)}): {moving_average:.2f} | "
                          f"{reward_details}")

        logger.info(reward_summary)

        self.agent.log_episode_reward(episode + 1, total_reward, moving_average)

        self.current_reward_types = {key: 0 for key in self.reward_weights}

        self.steps_since_last_attack = 0

    def run(self):
        """Run the main game loop."""
        if self.env.manual:
            start_listeners()
        logger.info("Press 'P' to start the screen capture")

        while self.agent.global_episode < self.env.episodes:
            episode = self.agent.global_episode
            logger.info(f"Starting Episode {episode + 1}")
            self.env.target_step = 0
            self.env.paused = pause_game(self.env.paused)
            game_window_img, screens = self.env.grab_screens()

            if game_window_img is None:
                logger.warning("Failed to capture screen, skipping iteration.")
                continue

            features = self.env.extract_features(screens)
            resized_img = self.env.resize_screen(game_window_img)
            state = self.env.prepare_state(resized_img)
            state_obj = GameState(features, state)

            while True:
                self.env.paused = pause_game(self.env.paused)

                action_mask = self.env.get_action_mask()

                if self.env.manual:
                    action = self.get_manual_action()
                    if action is None:
                        continue
                else:
                    action = self.agent.choose_action(state_obj.current_state, action_mask)

                if not self.env.manual and action is not None:
                    take_action(action, self.env.debugged, self.tool_manager)

                self.last_actions.append(action)

                game_window_img, screens = self.env.grab_screens()
                if game_window_img is None:
                    logger.warning("Failed to capture screen, skipping action.")
                    continue

                features = self.env.extract_features(screens)
                resized_img = self.env.resize_screen(game_window_img)
                next_state = self.env.prepare_state(resized_img)
                state_obj.update(features, next_state)

                self_hp = features['self_hp']
                boss_hp = features['boss_hp']

                current_time = time.time()
                if current_time - self.last_feature_log_time >= 1:
                    logger.info(f'Player Health: {self_hp:.2f}%, Boss Health: {boss_hp:.2f}%')
                    self.last_feature_log_time = current_time

                reward, self.defeated = self.action_judge(state_obj)

                if not self.env.manual:
                    if all(a == action for a in self.last_actions) and len(self.last_actions) == 8:
                        reward += self.reward_weights['idle_penalty']
                        self.current_reward_types['idle_penalty'] += self.reward_weights['idle_penalty']
                        logger.info("Idle penalty applied due to prolonged same activity.")

                if action is not None:
                    self.agent.store_transition(state_obj.current_state, action, reward, state_obj.next_state,
                                                self.defeated)

                self.env.target_step += 1
                if self.defeated:
                    break

            self.post_episode_updates(episode)
            self.agent.global_episode += 1

            restart(self.env, self.defeated)
            logger.info(f"Ending Episode {episode + 1}")

        cv2.destroyAllWindows()
        self.agent.close_writer()

    @staticmethod
    def get_manual_action():
        """Retrieve manual action from keyboard or mouse input."""
        if keyboard_result is not None:
            return keyboard_result
        elif mouse_result is not None:
            return mouse_result
        return None
