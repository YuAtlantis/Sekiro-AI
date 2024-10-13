import cv2
import time
import numpy as np
from game_control import take_action, pause_game, restart
from dueling_dqn import DQNAgent
from grab_screen import grab, extract_self_and_boss_blood

input_channels = 3
action_space = 2
model_file = ""


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood,
                 self_stop, boss_stop, emergence_break_mark):
    # Check if the game is over
    if next_self_blood < 5:
        reward = -10
        finish_flag = 1
        self_stop = 0
        boss_stop = 0
        emergence_break_mark = min(emergence_break_mark + 1, 100)
        print(f'You are dead and get the reward: {reward:.2f}')
        return reward, finish_flag, self_stop, boss_stop, emergence_break_mark

    if next_boss_blood < 8:
        reward = 20
        finish_flag = 0
        self_stop = 0
        boss_stop = 0
        emergence_break_mark = min(emergence_break_mark + 1, 100)
        print(f'You finally beat the boss and get the reward: {reward:.2f}')
        return reward, finish_flag, self_stop, boss_stop, emergence_break_mark

    # Calculate the change in health
    delta_self = next_self_blood - self_blood
    delta_boss = next_boss_blood - boss_blood

    print(f'Next moment self red: {next_self_blood:.2f}%, self current red: {self_blood:.2f} pixels')
    print(f'delta_self: {delta_self}')
    print(f'Next moment boss red: {next_boss_blood:.2f}%, boss current red: {boss_blood:.2f} pixels')
    print(f'delta_boss: {delta_boss}')

    # Initialize reward
    reward = 0

    # For player's health decrease
    if delta_self <= -6:
        if self_stop == 0:
            reward += -6
            self_stop = 1
    else:
        self_stop = 0

    # For boss's health decrease
    if delta_boss <= -3:
        if boss_stop == 0:
            reward += 4
            boss_stop = 1
    else:
        boss_stop = 0

    finish_flag = 0
    emergence_break_mark = 0
    print(f'Current reward is: {reward:.2f}')

    return reward, finish_flag, self_stop, boss_stop, emergence_break_mark


if __name__ == '__main__':
    paused = True
    width = 96
    height = 88
    episodes = 3000
    emergence_break = 0
    self_blood_window = (100, 650, 448, 663)
    boss_blood_window = (100, 180, 337, 195)

    agent = DQNAgent(input_channels, action_space, model_file)

    for episode in range(episodes):
        self_stop_mark = 0
        boss_stop_mark = 0
        target_step = 0

        print("Press t to start the screen capture")
        paused = pause_game(paused)
        self_screen = grab(self_blood_window)
        boss_screen = grab(boss_blood_window)

        self_blood, boss_blood = extract_self_and_boss_blood(self_screen, boss_screen)

        state = cv2.resize(self_screen, (width, height))
        last_time = time.time()
        while True:
            # Limit the frequency of grabbing the screen
            if time.time() - last_time < 0.5:
                continue

            state = np.array(state).reshape(input_channels, height, width)
            print('Screen -> State takes {} seconds in episode {}'.format(time.time() - last_time, episode))
            last_time = time.time()
            target_step += 1

            # Choose an action based on the current state
            action = agent.choose_action(state)

            # Take the selected action
            take_action(action)

            self_screen = grab(self_blood_window)
            boss_screen = grab(boss_blood_window)
            cv2.waitKey(1)

            next_state = cv2.resize(self_screen, (width, height))
            next_state = np.array(next_state).reshape(input_channels, height, width)

            next_self_blood, next_boss_blood = extract_self_and_boss_blood(self_screen, boss_screen)

            action_reward, done, self_stop_mark, boss_stop_mark, emergence_break = action_judge(boss_blood,
                                                                                                next_boss_blood,
                                                                                                self_blood,
                                                                                                next_self_blood,
                                                                                                self_stop_mark,
                                                                                                boss_stop_mark,
                                                                                                emergence_break)
            agent.store_transition(state, action, action_reward, next_state, done)

            # Update self_blood and boss_blood for the next iteration
            self_blood = next_self_blood
            boss_blood = next_boss_blood

            if done == 1:
                break

            paused = pause_game(paused)

        restart()

    cv2.destroyAllWindows()

