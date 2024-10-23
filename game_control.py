import time
import input_keys


def take_action(action_index, debugged, tool_manager):
    if not debugged:
        if action_index == 0:
            input_keys.defense()
        elif action_index == 1:
            input_keys.attack()
        elif action_index == 2:
            input_keys.tiptoe()
        elif action_index == 3:
            input_keys.jump()
        elif action_index in [4, 5, 6]:
            if not tool_manager.tools_exhausted:
                tool_manager.use_specific_tool(action_index - 4)
            else:
                print(f"Action {action_index} is no longer valid as tools are exhausted.")


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
            print("-------------------------You are dead and we are restarting a new round-------------------------")

        time.sleep(8)  # Wait for a period of time
        input_keys.clear_action_state()  # Clear the current input state
        input_keys.lock_vision()  # Lock the vision
        time.sleep(2)  # Wait for the vision to lock

        print("Please waiting 1s before taking further actions and restart the game...")
        time.sleep(1)  # Wait for some time to ensure action stability

        input_keys.attack()  # Simulate an attack action to restart the game
        if not boss_defeated:
            print("-------------------------A new round has already been started-------------------------")


def pause_game(paused):
    # Press T to start/stop the grab
    while True:
        keys = input_keys.key_check()
        if 'T' in keys:
            paused = not paused
            print('Game paused' if paused else 'Game started')
            wait_before_start(5, paused)

        if not paused:
            break

    return paused
