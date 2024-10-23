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
        # x, y, x_w, y_h
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

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


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
    MAX_SELF_RED_PIXELS = 2300
    MAX_BOSS_RED_PIXELS = 630

    # This will store the previous health values if they exist
    if not hasattr(extract_self_and_boss_blood, 'last_self_health'):
        extract_self_and_boss_blood.last_self_health = None
    if not hasattr(extract_self_and_boss_blood, 'last_boss_health'):
        extract_self_and_boss_blood.last_boss_health = None

    def calculate_health(screen, max_red_pixels, baseline_attr_name):
        current_red = count_red_pixels(screen)

        # Initialize or update the baseline red pixel count
        if not hasattr(extract_self_and_boss_blood, baseline_attr_name):
            setattr(extract_self_and_boss_blood, baseline_attr_name, current_red)
        else:
            baseline_red = getattr(extract_self_and_boss_blood, baseline_attr_name)
            setattr(extract_self_and_boss_blood, baseline_attr_name, max(baseline_red, current_red))

        # Cap the baseline red pixel count to the predefined maximum
        baseline_red = min(getattr(extract_self_and_boss_blood, baseline_attr_name), max_red_pixels)

        # Calculate the health percentage relative to the baseline
        if baseline_red > 0:
            health_percentage = (current_red / baseline_red) * 100
            return min(health_percentage, 100)  # Cap at 100%
        else:
            return 0

    # Calculate health percentages for both player and boss
    self_health_percentage = calculate_health(self_screen, MAX_SELF_RED_PIXELS, "self_baseline_red")
    boss_health_percentage = calculate_health(boss_screen, MAX_BOSS_RED_PIXELS, "boss_baseline_red")

    # Only print health info if it changed from the last recorded values
    if (self_health_percentage != extract_self_and_boss_blood.last_self_health or
            boss_health_percentage != extract_self_and_boss_blood.last_boss_health):
        print(f'Player Health: {self_health_percentage:.2f}%, Boss Health: {boss_health_percentage:.2f}%')

    # Update the last health values to the current ones
    extract_self_and_boss_blood.last_self_health = self_health_percentage
    extract_self_and_boss_blood.last_boss_health = boss_health_percentage

    # Display the health bar images for debugging
    cv2.imshow('Player Health Bar', self_screen)
    cv2.moveWindow('Player Health Bar', 100, 570)
    cv2.imshow('Boss Health Bar', boss_screen)
    cv2.moveWindow('Boss Health Bar', 100, 640)

    return self_health_percentage, boss_health_percentage


def extract_posture_bar(self_screen, boss_screen):
    # Define the HSV range for detecting the posture bar color
    LOWER_COLOR = np.array([0, 100, 100])
    UPPER_COLOR = np.array([30, 255, 255])

    # Initialize or check previous posture percentages
    if not hasattr(extract_posture_bar, 'last_self_posture'):
        extract_posture_bar.last_self_posture = None
    if not hasattr(extract_posture_bar, 'last_boss_posture'):
        extract_posture_bar.last_boss_posture = None

    # Function to calculate posture for a given screen
    def calculate_posture_percentage(screen):
        h, w = screen.shape[:2]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)

        # Create a mask for the defined color range
        mask_color = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

        # Focus on the middle section to ignore side decorations
        mid_section = mask_color[:, w // 4:3 * w // 4]
        color_ratio = 2 * cv2.countNonZero(mid_section) / (mid_section.shape[0] * mid_section.shape[1])

        # Perform edge detection
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (w * h)

        # Determine posture percentage based on thresholds
        if color_ratio > 0.1 and edge_ratio > 0.01:  # Adjust thresholds based on actual data
            return min(color_ratio, 1) * 100  # Convert to percentage and cap at 100%
        else:
            return 0  # No detectable posture bar

    # Calculate posture percentages for both player and boss
    self_posture_percentage = calculate_posture_percentage(self_screen)
    boss_posture_percentage = calculate_posture_percentage(boss_screen)

    # Only print posture info if it has changed since the last calculation
    if (self_posture_percentage != extract_posture_bar.last_self_posture or
            boss_posture_percentage != extract_posture_bar.last_boss_posture):
        print(f'Player Posture: {self_posture_percentage:.2f}%, Boss Posture: {boss_posture_percentage:.2f}%')

    # Update the last posture percentages
    extract_posture_bar.last_self_posture = self_posture_percentage
    extract_posture_bar.last_boss_posture = boss_posture_percentage

    # Display the posture bar images for debugging
    cv2.imshow('Player Posture Bar', self_screen)
    cv2.moveWindow('Player Posture Bar', 100, 710)
    cv2.imshow('Boss Posture Bar', boss_screen)
    cv2.moveWindow('Boss Posture Bar', 100, 780)

    return self_posture_percentage, boss_posture_percentage

