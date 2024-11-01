import time
import pygetwindow as gw
from keys import input_keys


def take_action(action_index, debugged, tool_manager):
    if not debugged:
        print(f"------action index:{action_index}")
        if action_index == 0:
            input_keys.defense()
        elif action_index == 1:
            input_keys.attack()
        elif action_index == 2:
            input_keys.tiptoe()
        elif action_index == 3:
            input_keys.jump()
        elif action_index == 4:
            input_keys.heal()
        elif action_index in [5, 6, 7]:
            tool_manager.use_specific_tool(action_index - 5)


def wait_before_start(seconds, paused):
    action = "stopping" if paused else "starting"
    print(f"Waiting for {seconds} seconds before {action}...")
    time.sleep(seconds)
    print("Game paused." if paused else "Game started.")


def focus_game_window():
    windows = gw.getWindowsWithTitle('Sekiro')
    if windows:
        game_window = windows[0]
        game_window.activate()
    else:
        print("Game window not found")


def restart(env, defeated, defeat_count):
    def reset_actions_and_pause():
        input_keys.clear_action_state()
        pause_game(True)

    def restart_sequence():
        print("-------------------------You are dead and we are restarting a new round-------------------------")
        input_keys.clear_action_state()
        print("-------------------------Waiting for 6 seconds before lock vision-------------------------")
        time.sleep(6)
        focus_game_window()
        time.sleep(1)
        input_keys.lock_vision()
        time.sleep(2)
        print("-------------------------Restart the fighting now by left click mouse-------------------------")
        input_keys.attack()
        time.sleep(2)

    if defeated == 1:
        if env.manual:
            print("-------------------------You are dead and should restart a new round in 4s-------------------------")
            time.sleep(4)
        else:
            restart_sequence()
    elif defeated == 2:
        print(f"-------------------------You beat the boss {defeat_count} times-------------------------")
        reset_actions_and_pause()


def pause_game(paused):
    while True:
        keys = input_keys.key_check()
        if 'T' in keys:
            paused = not paused
            print('Game paused' if paused else 'Game started')
            wait_before_start(3, paused)
        if paused:
            time.sleep(1)
        else:
            break

    return paused
