import time
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener, Key
import threading

# Global variables to store the result of mouse and keyboard inputs
mouse_result = 0
keyboard_result = 0

# Track the last mouse click and key press times
last_click_time = 0
last_keypress_time = 0
debounce_time = 0.08  # Set debounce time to 0.08 seconds


def on_click(x, y, button, pressed):
    global mouse_result, last_click_time
    if pressed:  # Only execute logic when the mouse is pressed
        current_time = time.time()
        # Execute if the time since the last click exceeds the debounce time
        if current_time - last_click_time > debounce_time:
            last_click_time = current_time  # Update last click time
            if button == button.left:
                mouse_result = 1  # Attack action
                print("Mouse Left Click - Attack, mouse_result:", mouse_result)
            elif button == button.right:
                mouse_result = 0  # Defense action
                print("Mouse Right Click - Defense, mouse_result:", mouse_result)
    return None


def on_press(key):
    global keyboard_result, last_keypress_time
    current_time = time.time()
    if current_time - last_keypress_time > debounce_time:
        last_keypress_time = current_time
        try:
            if hasattr(key, 'char') and key.char:
                print(f"Key pressed: {key.char}")
                keyboard_result = handle_key_action(key.char)  # Return the corresponding key action
                print(f"Action mapped for '{key.char}': keyboard_result:", keyboard_result)
        except AttributeError:
            print(f'Special Key {key} pressed')


# Keyboard release event handler
def on_release(key):
    if key == Key.esc:
        return False  # Stop the listener
    return True


# Tool management and action handling
tool_index = 1
tool_result = 3


def handle_key_action(key_char):
    global tool_index
    global tool_result

    action_mapping = {
        'e': (2, "tiptoe"),
        'a': (3, "left"),
    }

    if key_char == 'z':  # Tool switch
        tool_index += 1
        if tool_index > 3:
            tool_index = 1
        tool_result = 3 + tool_index
        print(f"Change to the game tool {tool_index}")

    elif key_char == '3':  # Use the tool
        tool_result = 3 + tool_index
        print(f"Use the game tool {tool_index}, return {tool_result}")
        return tool_result

    elif key_char in action_mapping:
        result, action = action_mapping[key_char]
        return result


# Start mouse and keyboard listeners
def start_listeners():
    # Start mouse and keyboard listeners in threads to prevent blocking the main process
    mouse_listener = MouseListener(on_click=on_click)
    keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)

    # Set as daemon threads, so they automatically exit when the main process ends
    mouse_listener_thread = threading.Thread(target=mouse_listener.start, daemon=True)
    keyboard_listener_thread = threading.Thread(target=keyboard_listener.start, daemon=True)

    # Start the listener threads
    mouse_listener_thread.start()
    keyboard_listener_thread.start()

    # Return the thread objects so the main program can continue running other tasks
    return mouse_listener_thread, keyboard_listener_thread


if __name__ == "__main__":
    mouse_thread, keyboard_thread = start_listeners()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program.")

