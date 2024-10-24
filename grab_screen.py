import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
import logging
import pytesseract

logging.basicConfig(level=logging.INFO)


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

    def calculate_health(screen, max_red_pixels, baseline_attr_name):
        current_red = count_red_pixels(screen)

        # Initialize or update the baseline red pixel count
        if not hasattr(calculate_health, baseline_attr_name):
            setattr(calculate_health, baseline_attr_name, current_red)
        else:
            baseline_red = getattr(calculate_health, baseline_attr_name)
            setattr(calculate_health, baseline_attr_name, max(baseline_red, current_red))

        # Cap the baseline red pixel count to the predefined maximum
        baseline_red = min(getattr(calculate_health, baseline_attr_name), max_red_pixels)

        # Calculate the health percentage relative to the baseline
        if baseline_red > 0:
            health_percentage = (current_red / baseline_red) * 100
            return min(health_percentage, 100)  # Cap at 100%
        else:
            return 0

    # Calculate health percentages for both player and boss
    self_health_percentage = calculate_health(self_screen, MAX_SELF_RED_PIXELS, "self_baseline_red")
    boss_health_percentage = calculate_health(boss_screen, MAX_BOSS_RED_PIXELS, "boss_baseline_red")

    logging.info(f'Player Health: {self_health_percentage:.2f}%, Boss Health: {boss_health_percentage:.2f}%')

    # Display the health bar images for debugging if needed
    cv2.imshow('Player Health Bar', self_screen)
    cv2.moveWindow('Player Health Bar', 100, 570)
    cv2.imshow('Boss Health Bar', boss_screen)
    cv2.moveWindow('Boss Health Bar', 100, 640)

    return self_health_percentage, boss_health_percentage


def extract_posture_bar(self_screen, boss_screen):
    # Define the HSV range for detecting the posture bar color
    LOWER_COLOR = np.array([0, 100, 100])
    UPPER_COLOR = np.array([30, 255, 255])

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

    logging.info(f'Player Posture: {self_posture_percentage:.2f}%, Boss Posture: {boss_posture_percentage:.2f}%')

    # Display the posture bar images for debugging if needed
    cv2.imshow('Player Posture Bar', self_screen)
    cv2.moveWindow('Player Posture Bar', 100, 710)
    cv2.imshow('Boss Posture Bar', boss_screen)
    cv2.moveWindow('Boss Posture Bar', 100, 780)

    return self_posture_percentage, boss_posture_percentage


def get_remaining_uses(region, current_remaining_uses):
    # Grab the screen region
    screenshot = grab(region)

    # Convert to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract text using OCR
    config = "--psm 7"
    extracted_text = pytesseract.image_to_string(binary, config=config, lang='eng').strip()

    # Error handling to ensure extracted text is a valid number
    try:
        remaining_uses = int(extracted_text)
        # If successfully extracted, update the remaining uses
        current_remaining_uses = remaining_uses
    except ValueError:
        pass
        # Do not change current_remaining_uses if extraction fails

    # Display the original screenshot and processed image, adjusted to fit the window size
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', screenshot.shape[1], screenshot.shape[0])
    cv2.moveWindow('Original Image', 550, 535)
    cv2.imshow('Original Image', screenshot)

    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Image', binary.shape[1], binary.shape[0])
    cv2.moveWindow('Processed Image', 550, 755)
    cv2.imshow('Processed Image', binary)

    # Return the updated current_remaining_uses
    return current_remaining_uses



