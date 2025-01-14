import ctypes
import time
import win32api as wapi
from .keys_dictionary import KEY_CODES
from .keys_dictionary import MOUSE_CODES

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


def left_click():
    mouse_action(MOUSE_CODES['LEFT_CLICK'])
    time.sleep(0.05)
    mouse_action(MOUSE_CODES['LEFT_RELEASE'])


def right_click():
    mouse_action(MOUSE_CODES['RIGHT_CLICK'])
    time.sleep(0.05)
    mouse_action(MOUSE_CODES['RIGHT_RELEASE'])


def middle_click():
    mouse_action(MOUSE_CODES['MIDDLE_CLICK'])
    time.sleep(0.05)
    mouse_action(MOUSE_CODES['MIDDLE_RELEASE'])


def move_mouse(dx, dy):
    mouse_action(0x0001, dx, dy)


pressed_keys = set()
pressed_mouse_buttons = set()


def press_key(hexKeyCode, press_time=0.05):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    pressed_keys.add(hexKeyCode)
    time.sleep(press_time)


def release_key(hexKeyCode):
    if hexKeyCode in pressed_keys:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        pressed_keys.remove(hexKeyCode)


def mouse_action(mouseCode, dx=0, dy=0):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(dx, dy, 0, mouseCode, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    if mouseCode in [MOUSE_CODES['LEFT_CLICK'], MOUSE_CODES['RIGHT_CLICK'], MOUSE_CODES['MIDDLE_CLICK']]:
        pressed_mouse_buttons.add(mouseCode)
    elif mouseCode in [MOUSE_CODES['LEFT_RELEASE'], MOUSE_CODES['RIGHT_RELEASE'], MOUSE_CODES['MIDDLE_RELEASE']]:
        click_code = mouseCode - 2
        if click_code in pressed_mouse_buttons:
            pressed_mouse_buttons.remove(click_code)


def perform_action(action, duration):
    if action in KEY_CODES:
        press_key(KEY_CODES[action], duration)
        time.sleep(duration)
        release_key(KEY_CODES[action])


def defense():
    right_click()


def attack():
    left_click()


def jump():
    perform_action('SPACE', 0.1)


def tiptoe():
    perform_action("E", 0.1)


def heal():
    perform_action("1", 0.05)


def lock_vision():
    middle_click()


def press_esc():
    perform_action('ESC', 0.3)


def backward_dodge():
    press_key(KEY_CODES['S'], 0.3)
    tiptoe()
    release_key(KEY_CODES['S'])


def left():
    perform_action('A', 0.1)


def right_dodge():
    press_key(KEY_CODES['D'], 0.3)
    tiptoe()
    release_key(KEY_CODES['D'])


keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def clear_action_state():
    for key in list(pressed_keys):
        release_key(key)
    for mouse_code in list(pressed_mouse_buttons):
        release_code = mouse_code + 2
        mouse_action(release_code)
