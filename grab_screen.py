import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


# Use Windows API to capture the screen
def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img[..., :3]  # Return only RGB channels


# Calculate the number of red pixels in an image
def count_red_pixels(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red color (two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Count red pixels
    red_pixels = cv2.countNonZero(mask)

    return red_pixels


# Main loop: real-time capture and health detection
def main():
    wait_time = 3  # Reduced wait time for faster startup
    # Define regions for player and Boss health bars
    self_blood_window = (100, 650, 448, 663)
    boss_blood_window = (100, 180, 337, 195)

    # Wait for initialization
    for i in list(range(wait_time))[::-1]:
        print(f"Program will start in {i + 1} seconds")
        time.sleep(1)

    # Capture the initial screen regions to get the initial baseline red pixel counts
    self_screen = grab_screen(self_blood_window)
    boss_screen = grab_screen(boss_blood_window)

    # Get the initial baseline red pixel count
    self_baseline_red = count_red_pixels(self_screen)
    boss_baseline_red = count_red_pixels(boss_screen)

    while True:
        # Capture screen regions
        self_screen = grab_screen(self_blood_window)
        boss_screen = grab_screen(boss_blood_window)

        # Calculate current red pixel count
        self_current_red = count_red_pixels(self_screen)
        boss_current_red = count_red_pixels(boss_screen)
        print(self_baseline_red)
        print(self_current_red)

        # Update the baseline if current red pixel count is significantly higher (indicating health increase)
        if self_baseline_red < self_current_red < 2000:  # Use a threshold to avoid false positives due to noise
            self_baseline_red = self_current_red

        if boss_baseline_red < boss_current_red < 635:
            boss_baseline_red = boss_current_red

        # Calculate the health percentage relative to the baseline
        if self_baseline_red > 0:
            self_health_percentage = (self_current_red / self_baseline_red) * 100
        else:
            self_health_percentage = 0

        if boss_baseline_red > 0:
            boss_health_percentage = (boss_current_red / boss_baseline_red) * 100
        else:
            boss_health_percentage = 0

        # Output health information
        print(f'Player Health: {self_health_percentage:.2f}%, Boss Health: {boss_health_percentage:.2f}%')

        # Display the health bar images for debugging
        cv2.imshow('Player Health Bar', self_screen)
        cv2.imshow('Boss Health Bar', boss_screen)

        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()