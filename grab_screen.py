import cv2
import numpy as np
import mss
import logging
import pytesseract

logging.basicConfig(level=logging.INFO)


# Capture the full screen using mss
def grab_full_screen(region=(0, 0, 1024, 620)):
    # Capture the entire screen
    with mss.mss() as sct:
        monitor = {
            "top": region[1],
            "left": region[0],
            "width": region[2] - region[0],
            "height": region[3] - region[1]
        }
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        return img[:, :, :3]


def grab_region(full_screen, region):
    # Extract a specific region from the full screen image
    x1, y1, x2, y2 = region
    return full_screen[y1:y2 + 1, x1:x2 + 1]


# Extract health percentages for the player and the Boss
def extract_health(self_screen, boss_screen):
    # Function to calculate health percentage based on health bar length
    def calculate_health_percentage(health_bar_image):
        # Convert to HSV color space
        hsv = cv2.cvtColor(health_bar_image, cv2.COLOR_BGR2HSV)

        # Adjusted red color range for the health bar
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Use fixed threshold
        fixed_threshold = 15  # Adjust this value based on experimentation
        _, binary_mask = cv2.threshold(mask, fixed_threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the health bar
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            health_percentage = (w / binary_mask.shape[1]) * 100
            health_percentage = np.clip(health_percentage, 0, 100)
        else:
            health_percentage = 0

        return health_percentage

    # Calculate health percentages for the player and the Boss
    self_health = calculate_health_percentage(self_screen)
    boss_health = calculate_health_percentage(boss_screen)

    logging.info(f'Player Health: {self_health:.2f}%, Boss Health: {boss_health:.2f}%')

    # Optionally display health bar images for debugging
    cv2.imshow('Player Health Bar', self_screen)
    cv2.moveWindow('Player Health Bar', 100, 640)
    cv2.imshow('Boss Health Bar', boss_screen)
    cv2.moveWindow('Boss Health Bar', 100, 710)

    return self_health, boss_health


# Extract posture percentages for the player and the Boss
def extract_posture(self_screen, boss_screen):
    # Function to calculate posture percentage based on posture bar length
    def calculate_posture_percentage(posture_bar_image):
        # Convert to HSV color space
        hsv = cv2.cvtColor(posture_bar_image, cv2.COLOR_BGR2HSV)

        # Define color range for the posture bar (adjusted for yellow/orange)
        lower_color = np.array([15, 100, 100])
        upper_color = np.array([30, 255, 255])

        # Create posture bar color mask
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Use Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Project the mask along the vertical axis to get a horizontal profile
        profile = np.sum(mask, axis=0)  # Sum along height to get values along the width

        # Threshold the profile to detect the posture bar length
        threshold = np.max(profile) * 0.3  # Use 50% of the max value as the threshold
        indices = np.where(profile > threshold)[0]

        if len(indices) > 0:
            posture_bar_length = indices[-1] - indices[0]
            posture_percentage = (posture_bar_length / mask.shape[1]) * 100
            posture_percentage = np.clip(posture_percentage, 0, 100)
        else:
            posture_percentage = 0  # No posture bar detected

        return posture_percentage

    # Calculate posture percentages for the player and the Boss
    self_posture = calculate_posture_percentage(self_screen)
    boss_posture = calculate_posture_percentage(boss_screen)

    logging.info(f'Player Posture: {self_posture:.2f}%, Boss Posture: {boss_posture:.2f}%')

    # Optionally display posture bar images for debugging
    cv2.namedWindow('Player Posture Bar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Player Posture Bar', self_screen.shape[1], self_screen.shape[0])
    cv2.moveWindow('Player Posture Bar', 100, 780)
    cv2.imshow('Player Posture Bar', self_screen)

    cv2.namedWindow('Boss Posture Bar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Boss Posture Bar', boss_screen.shape[1], boss_screen.shape[0])
    cv2.moveWindow('Boss Posture Bar', 100, 890)
    cv2.imshow('Boss Posture Bar', boss_screen)

    return self_posture, boss_posture


# Extract remaining uses of items
def get_remaining_uses(screenshot, current_remaining):
    # Convert to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use OCR to extract text
    config = "--psm 7"
    extracted_text = pytesseract.image_to_string(binary, config=config, lang='eng').strip()

    # Error handling to ensure the extracted text is a valid number
    try:
        remaining_uses = int(extracted_text)
        # Update the remaining uses if extraction is successful
        current_remaining = remaining_uses
    except ValueError:
        pass

    # Display the original and processed images, adjust to fit window size
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', screenshot.shape[1], screenshot.shape[0])
    cv2.moveWindow('Original Image', 550, 645)
    cv2.imshow('Original Image', screenshot)

    cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Image', binary.shape[1], binary.shape[0])
    cv2.moveWindow('Processed Image', 550, 825)
    cv2.imshow('Processed Image', binary)

    # Return the updated current_remaining_uses
    return current_remaining


if __name__ == "__main__":
    regions = {
        'self_blood': (54, 562, 400, 576),
        'boss_blood': (57, 93, 288, 105),
        'self_posture': (395, 535, 635, 552),
        'boss_posture': (315, 73, 710, 88),
        'remaining_uses': (955, 570, 970, 588)
    }

    current_remaining_uses = 19

    while True:
        # Capture the full screen once
        full_screen_img = grab_full_screen()

        # Extract all necessary regions at once
        screens = {name: grab_region(full_screen_img, region) for name, region in regions.items()}

        # Extract health percentages for the player and the Boss
        self_health_percentage, boss_health_percentage = extract_health(
            screens['self_blood'], screens['boss_blood'])

        # Extract posture percentages for the player and the Boss
        self_posture_percentage, boss_posture_percentage = extract_posture(
            screens['self_posture'], screens['boss_posture'])

        # Get remaining uses
        current_remaining_uses = get_remaining_uses(screens['remaining_uses'], current_remaining_uses)

        # Keep the window open until any key is pressed
        cv2.waitKey(1)
