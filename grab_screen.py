import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui


# Use Windows API to capture the screen
def grab(region):
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


def extract_self_and_boss_blood(self_screen, boss_screen):
    # Maximum possible red pixel counts for player and boss health bars
    MAX_SELF_RED_PIXELS = 2300  # Adjust this value based on actual max pixels
    MAX_BOSS_RED_PIXELS = 630   # Adjust this value based on actual max pixels

    # Calculate current red pixel count
    self_current_red = count_red_pixels(self_screen)
    boss_current_red = count_red_pixels(boss_screen)

    # Initialize or update the baseline red pixel counts
    if not hasattr(extract_self_and_boss_blood, "self_baseline_red"):
        extract_self_and_boss_blood.self_baseline_red = self_current_red
    else:
        extract_self_and_boss_blood.self_baseline_red = max(extract_self_and_boss_blood.self_baseline_red, self_current_red)

    if not hasattr(extract_self_and_boss_blood, "boss_baseline_red"):
        extract_self_and_boss_blood.boss_baseline_red = boss_current_red
    else:
        extract_self_and_boss_blood.boss_baseline_red = max(extract_self_and_boss_blood.boss_baseline_red, boss_current_red)

    # Cap the baseline red pixel counts to the predefined maximums
    self_baseline_red = min(extract_self_and_boss_blood.self_baseline_red, MAX_SELF_RED_PIXELS)
    boss_baseline_red = min(extract_self_and_boss_blood.boss_baseline_red, MAX_BOSS_RED_PIXELS)

    print(f'Player current red: {self_current_red:.2f} pixels, Player baseline red: {self_baseline_red:.2f} pixels')
    print(f'Boss current red: {boss_current_red:.2f} pixels, Boss baseline red: {boss_baseline_red:.2f} pixels')

    # Calculate the health percentage relative to the baseline
    if self_baseline_red > 0:
        self_health_percentage = (self_current_red / self_baseline_red) * 100
        self_health_percentage = min(self_health_percentage, 100)  # Cap at 100%
    else:
        self_health_percentage = 0

    if boss_baseline_red > 0:
        boss_health_percentage = (boss_current_red / boss_baseline_red) * 100
        boss_health_percentage = min(boss_health_percentage, 100)  # Cap at 100%
    else:
        boss_health_percentage = 0

    # Output health information
    print(f'Player Health: {self_health_percentage:.2f}%, Boss Health: {boss_health_percentage:.2f}%')

    # Display the health bar images for debugging
    cv2.imshow('Player Health Bar', self_screen)
    cv2.imshow('Boss Health Bar', boss_screen)

    return self_health_percentage, boss_health_percentage

