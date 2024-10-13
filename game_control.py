import time
import input_keys


def take_action(action_index):
    if action_index == 0:
        input_keys.defense()
    elif action_index == 1:
        input_keys.attack()
    elif action_index == 2:
        input_keys.tiptoe()
    elif action_index == 3:
        input_keys.jump()


def wait_before_start(seconds, paused):
    action = "stopping" if paused else "starting"
    print(f"Waiting for {seconds} seconds before {action}...")
    time.sleep(seconds)
    print("Game paused." if paused else "Game started.")


def restart():
    print("----------You dead,restart a new round----------")
    time.sleep(8)
    input_keys.lock_vision()
    time.sleep(0.2)
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