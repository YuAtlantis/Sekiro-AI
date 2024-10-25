import time
import input_keys


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
            input_keys.left_dodge()
        elif action_index == 4:
            input_keys.heal()
        elif action_index in [5, 6, 7]:
            tool_manager.use_specific_tool(action_index - 5)


def wait_before_start(seconds, paused):
    action = "stopping" if paused else "starting"
    print(f"Waiting for {seconds} seconds before {action}...")
    time.sleep(seconds)
    print("Game paused." if paused else "Game started.")


def restart(debugged, defeated):
    if not debugged:
        if defeated == 1:
            print("-------------------------You are dead and we are restarting a new round-------------------------")
            input_keys.clear_action_state()
            print("-------------------------Waiting for 8 seconds before lock vision-------------------------")
            time.sleep(8)
            print("-------------------------Waiting for 2 seconds for locking vision-------------------------")
            time.sleep(1)
            input_keys.lock_vision()
            time.sleep(1)
            print("-------------------------Restart the fighting now by left click mouse-------------------------")
            input_keys.attack()  # Simulate an attack action to restart the game

        elif defeated == 2:
            time.sleep(2)
            input_keys.attack()
            time.sleep(1)
            input_keys.attack()
            print("-------------------------You beat the boss and wait for the animation-------------------------")
            input_keys.clear_action_state()  # Clear the current input state
            print("-------------------------Waiting for 2 seconds for locking vision-------------------------")
            time.sleep(2)
            input_keys.lock_vision()


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
