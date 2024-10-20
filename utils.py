import cv2
from grab_screen import grab

# ---------- Parameters to be set ----------
GAME_WIDTH = 1024  # Game window width
GAME_HEIGHT = 576  # Game window height
WHITE_BORDER = 40  # Game border


def get_game_screen():
    region = [0, WHITE_BORDER, GAME_WIDTH, GAME_HEIGHT + WHITE_BORDER]
    return grab(region)


if __name__ == "__main__":
    screen = get_game_screen()
    cv2.imshow('screen', screen)
    cv2.waitKey(0)
