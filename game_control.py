import time
import input_keys


def take_action(action_index, debugged):
    if not debugged:
        if action_index == 0:
            input_keys.defense()
        elif action_index == 1:
            input_keys.attack()
        elif action_index == 2:
            input_keys.tiptoe()


def wait_before_start(seconds, paused):
    action = "stopping" if paused else "starting"
    print(f"Waiting for {seconds} seconds before {action}...")
    time.sleep(seconds)
    print("Game paused." if paused else "Game started.")


def restart(debugged):
    if not debugged:
        print("----------You dead, restart a new round----------")
        time.sleep(8.5)
        input_keys.clear_action_state()
        input_keys.lock_vision()
        time.sleep(1)
        print("Waiting before taking further actions...")
        time.sleep(2)
        input_keys.attack()
        print("----------A new round has already started----------")


def pause_game(paused):
    # Press t to start/stop the grab
    while True:
        keys = input_keys.key_check()
        if 'T' in keys:
            paused = not paused
            print('Game paused' if paused else 'Game started')
            wait_before_start(3, paused)

        if not paused:
            break

    return paused
