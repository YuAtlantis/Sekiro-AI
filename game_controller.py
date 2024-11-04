# game_controller.py

import cv2
import logging
import time
from game_environment import GameEnvironment
from game_agent import GameAgent
from game_state import GameState
from control.tool_manager import ToolManager
from keys.input_keys import attack
from control.dueling_dqn_manual import keyboard_result, mouse_result, start_listeners
from control.game_control import take_action, pause_game, restart
from logging.handlers import RotatingFileHandler

# Configure logging with rotating file handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('./logs/game_controller.log', maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GameController:
    def __init__(self):
        self.total_reward = 0

        self.defeated = 0
        self.defeat_count = 0

        self.missing_boss_hp_steps = 0
        self.boss_lives = 1

        self.successful_defense_count = 0
        self.max_defense_rewards = 5

        self.steps_since_last_attack = 0
        self.idle_threshold = 25

        self.time_penalty_increment = -0.001

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
            'self_hp_loss': -0.4,
            'boss_hp_loss': 1.0,
            'self_death': -20,
            'self_posture_increase': -0.15,
            'boss_posture_increase': 0.3,
            'defeat_bonus': 30,
            'time_penalty': -0.1,
            'successful_defense': 1.0,
            "intermediate_defeat": 0,
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
        """Judge the action and calculate the reward."""
        reward, defeated = 0, 0
        self.defeat_window_start = None

        # 1. Apply Time Penalty
        time_penalty = self.reward_weights['time_penalty'] + (self.env.target_step * self.time_penalty_increment)
        reward += time_penalty
        self.current_reward_types['time_penalty'] += time_penalty

        # 2. Handle Successful Defense
        if self.check_successful_defense(state_obj):
            if self.successful_defense_count < self.max_defense_rewards:
                defense_reward = self.reward_weights['successful_defense']
                reward += defense_reward
                self.current_reward_types['successful_defense'] += defense_reward
                self.successful_defense_count += 1
                logger.info(f"Successful defense #{self.successful_defense_count} rewarded.")
            else:
                logger.info("Maximum number of successful defenses reached; no additional rewards.")

        # 3. Apply Idle Penalty
        if self.steps_since_last_attack >= self.idle_threshold:
            idle_penalty = -2
            reward += idle_penalty
            self.current_reward_types['idle_penalty'] += idle_penalty
            logger.info("Idle penalty applied due to prolonged inactivity.")
            self.steps_since_last_attack = 0

        # 4. Check for Death
        if state_obj.next_features['self_hp'] <= 1:
            death_penalty = self.reward_weights['self_death']
            reward += death_penalty
            self.current_reward_types['self_death'] += death_penalty
            defeated = 1
            logger.info("Agent has died. Death penalty applied.")
            self.total_reward += reward
            return reward, defeated

        # 5. Check for Boss Defeat
        if state_obj.next_features['boss_hp'] <= 1:
            defeat_reward, defeated = self.handle_boss_low_health(state_obj)
            reward += defeat_reward
        else:
            # 6. Calculate Deltas and Corresponding Rewards
            delta_reward = self.calculate_deltas(state_obj)
            reward += delta_reward

        self.total_reward += reward
        return reward, defeated

    @staticmethod
    def check_successful_defense(state_obj):
        """Check if a successful defense occurred."""
        hp_delta = state_obj.next_features['self_hp'] - state_obj.current_features['self_hp']
        posture_delta = state_obj.next_features['self_posture'] - state_obj.current_features['self_posture']
        return hp_delta < 1 and posture_delta > 0

    def handle_boss_low_health(self, state_obj):
        """Handle the scenario when the boss's health is low."""
        if self.boss_lives == 1:
            # Boss has only one life; attempt to defeat immediately
            logger.info("Boss has a single life. Attempting to defeat immediately.")
            reward = self.attack_directly()
            self.defeated = 2  # Indicate boss defeated
            return reward, self.defeated
        else:
            # Boss has multiple lives; proceed with existing defeat window logic
            if not self.defeat_window_start:
                self.defeat_window_start = time.time()

            logger.info("Boss HP <= 1, initiating defeat protocol.")

            reward = self.attack_in_low_health_phase(state_obj)

            if self.missing_boss_hp_steps > 40:
                defeat_bonus = self.reward_weights.get('defeat_bonus', 40)
                reward += defeat_bonus
                self.defeated = 2
                logger.info("Boss HP missing steps exceeded threshold; game will stop.")
            else:
                self.defeated = 0

            return reward, self.defeated

    def attack_directly(self):
        """Attack the boss directly to defeat it."""
        attack()  # Perform the attack using the imported function
        defeat_bonus = self.reward_weights.get('defeat_bonus', 40)
        reward = defeat_bonus
        self.current_reward_types['defeat_bonus'] += defeat_bonus
        logger.info("Boss defeated directly; defeat bonus awarded.")
        return reward

    def attack_in_low_health_phase(self, state_obj):
        """Attack during the boss's low health phase."""
        reward = 0
        time_elapsed = time.time() - self.defeat_window_start if self.defeat_window_start else 0

        if state_obj.next_features['boss_hp'] <= 0:
            self.missing_boss_hp_steps += 1
        else:
            self.missing_boss_hp_steps = 0

        logger.info(f"Missing Boss HP Steps: {self.missing_boss_hp_steps}")

        if state_obj.current_features['boss_hp'] > 50:
            reward += self.reward_weights['defeat_bonus']
            self.current_reward_types['defeat_bonus'] += self.reward_weights['defeat_bonus']
            self.defeat_count += 1
            self.defeat_window_start = None
            logger.info(f"Boss HP restored above 50%; entering next phase: {self.defeat_count}")
        else:
            if time_elapsed < 8:
                attack()
                logger.info("Continuing to attack Boss to ensure defeat...")
            else:
                self.defeat_window_start = None

        return reward

    def calculate_deltas(self, state_obj):
        """Calculate the reward based on the changes in features."""
        keys = ['self_hp', 'boss_hp', 'self_posture', 'boss_posture']
        deltas = {key: state_obj.next_features[key] - state_obj.current_features[key] for key in keys}
        reward = 0

        # 1. Self HP loss penalty
        if deltas['self_hp'] < 0:
            self_hp_loss = self.reward_weights['self_hp_loss'] * abs(deltas['self_hp'])
            reward += self_hp_loss
            self.current_reward_types['self_hp_loss'] += self_hp_loss
        else:
            self_hp_loss = 0

        # 2. Boss HP loss reward
        if deltas['boss_hp'] < 0:
            boss_hp_reward = self.reward_weights['boss_hp_loss'] * abs(deltas['boss_hp'])
            reward += boss_hp_reward
            self.current_reward_types['boss_hp_loss'] += boss_hp_reward
            self.steps_since_last_attack = 0  # Reset idle counter on successful attack
        else:
            boss_hp_reward = 0

        # 3. Intermediate rewards based on boss HP thresholds
        boss_hp_percentage = state_obj.next_features['boss_hp']
        if 0.75 > boss_hp_percentage >= 0.5 and not self.intermediate_rewards_given['75%']:
            reward += 3
            self.current_reward_types['intermediate_defeat'] += 3
            self.intermediate_rewards_given['75%'] = True
            logger.info("Intermediate reward granted for boss HP between 50% and 75%.")
        elif 0.5 > boss_hp_percentage >= 0.25 and not self.intermediate_rewards_given['50%']:
            reward += 7
            self.current_reward_types['intermediate_defeat'] += 7
            self.intermediate_rewards_given['50%'] = True
            logger.info("Intermediate reward granted for boss HP between 25% and 50%.")
        elif boss_hp_percentage < 0.25 and not self.intermediate_rewards_given['25%']:
            reward += 12
            self.current_reward_types['intermediate_defeat'] += 12
            self.intermediate_rewards_given['25%'] = True
            logger.info("Intermediate reward granted for boss HP below 25%.")

        # 4. Self posture increase penalty
        if deltas['self_posture'] > 0 and state_obj.current_features['self_posture'] > 80:
            self_posture_penalty = self.reward_weights['self_posture_increase'] * deltas['self_posture']
            reward += self_posture_penalty
            self.current_reward_types['self_posture_increase'] += self_posture_penalty
            logger.info(f"Self posture increased by {deltas['self_posture']:.2f}; penalty applied.")
        else:
            self_posture_penalty = 0

        # 5. Boss posture increase reward
        if deltas['boss_posture'] > 0:
            boss_posture_reward = self.reward_weights['boss_posture_increase'] * deltas['boss_posture']
            reward += boss_posture_reward
            self.current_reward_types['boss_posture_increase'] += boss_posture_reward
            logger.info(f"Boss posture increased by {deltas['boss_posture']:.2f}; reward applied.")
        else:
            boss_posture_reward = 0

        # Log reward details
        if self.env.target_step % 10 == 0:
            logger.info(
                f"Step {self.env.target_step}: Reward Details - "
                f"Self HP Loss Penalty: {self_hp_loss:.2f}, "
                f"Boss HP Loss Reward: {boss_hp_reward:.2f}, "
                f"Intermediate Defeat Reward: {reward - self_hp_loss - boss_hp_reward - self_posture_penalty - boss_posture_reward:.2f}, "
                f"Self Posture Increase Penalty: {self_posture_penalty:.2f}, "
                f"Boss Posture Increase Reward: {boss_posture_reward:.2f}"
            )

        return reward

    def post_episode_updates(self, episode):
        """Update statistics and save models after each episode."""
        for key in self.reward_type_distribution:
            self.reward_type_distribution[key].append(self.current_reward_types.get(key, 0))

        reward_details = " | ".join([f"{key}: {value:.2f}" for key, value in self.current_reward_types.items()])
        reward_summary = f"Episode {episode + 1} Summary: Total Reward: {self.total_reward:.2f} | {reward_details}"

        logger.info(reward_summary)

        # Reset total reward and current reward types
        self.total_reward = 0
        self.current_reward_types = {key: 0 for key in self.reward_weights}

        # Reset defense count and intermediate rewards
        self.successful_defense_count = 0
        self.intermediate_rewards_given = {
            '75%': False,
            '50%': False,
            '25%': False
        }

        # Reset idle steps
        self.steps_since_last_attack = 0

        # Save model every 50 episodes
        if episode % 50 == 0 and not self.env.debugged:
            self.agent.save_model(episode)

    def run(self):
        """Run the main game_settings loop."""
        if self.env.manual:
            start_listeners()
        logger.info("Press 'P' to start the screen capture")
        for episode in range(self.env.episodes):
            self.env.reset_marks()
            self.env.paused = pause_game(self.env.paused)
            game_window_img, screens, remaining_uses_img = self.env.grab_screens()

            if game_window_img is None:
                logger.warning("Failed to capture screen, skipping iteration.")
                continue

            features = self.env.extract_features(screens)
            resized_img = self.env.resize_screen(game_window_img)
            state = self.env.prepare_state(resized_img)
            state_obj = GameState(features, state)

            while True:
                self.env.target_step += 1
                self.env.paused = pause_game(self.env.paused)

                action_mask = self.env.get_action_mask()

                if self.env.manual:
                    action = self.get_manual_action()
                else:
                    action = self.agent.choose_action(state_obj.current_state, action_mask)

                if not self.env.manual and action is not None:
                    take_action(action, self.env.debugged, self.tool_manager)

                game_window_img, screens, remaining_uses_img = self.env.grab_screens()
                if game_window_img is None:
                    logger.warning("Failed to capture screen, skipping action.")
                    continue

                if self.env.target_step % 100 == 0:
                    self.env.update_remaining_uses(remaining_uses_img)

                features = self.env.extract_features(screens)
                resized_img = self.env.resize_screen(game_window_img)
                next_state = self.env.prepare_state(resized_img)
                state_obj.update(features, next_state)

                reward, self.defeated = self.action_judge(state_obj)

                if action is not None:
                    self.agent.store_transition(state_obj.current_state, action, reward, state_obj.next_state,
                                                self.defeated)

                if self.env.target_step % 64 == 0:
                    self.agent.train()

                if self.defeated:
                    break

            self.post_episode_updates(episode)
            restart(self.env, self.defeated, self.defeat_count)

        cv2.destroyAllWindows()

    @staticmethod
    def get_manual_action():
        """Retrieve manual action from keyboard or mouse input."""
        if keyboard_result is not None:
            return keyboard_result
        elif mouse_result is not None:
            return mouse_result
        return None
