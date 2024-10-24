import time
import input_keys
from threading import Lock


def take_action(action_index, debugged, tool_manager):
    if not debugged:
        print(f"------action index:{action_index}")
        if action_index == 0:
            input_keys.defense()
        elif action_index == 1:
            input_keys.attack()
        elif action_index == 2:
            input_keys.jump()
        elif action_index == 3:
            input_keys.tiptoe()
        elif action_index == 4:
            input_keys.left_dodge()
        elif action_index == 5:
            input_keys.heal()
        elif action_index in [6, 7, 8]:
            tool_manager.use_specific_tool(action_index - 6)


def wait_before_start(seconds, paused):
    action = "stopping" if paused else "starting"
    print(f"Waiting for {seconds} seconds before {action}...")
    time.sleep(seconds)
    print("Game paused." if paused else "Game started.")


def restart(debugged, boss_defeated=False):
    if not debugged:
        if boss_defeated:
            print("-------------------------Boss is defeated and training is finished-------------------------")
            return
        else:
            print(
                "-------------------------You are dead and we are restarting a new round-------------------------")

        input_keys.clear_action_state()  # Clear the current input state
        time.sleep(5)  # Wait for a period of time
        input_keys.lock_vision()  # Lock the vision
        time.sleep(2)  # Wait for the vision to lock
        input_keys.attack()  # Simulate an attack action to restart the game

        if not boss_defeated:
            print("-------------------------A new round has already been started-------------------------")


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
