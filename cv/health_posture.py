# health_posture.py

import cv2
import numpy as np
import logging
from numba import njit
from collections import deque

logging.basicConfig(level=logging.INFO)

DEBUG_MODE = False  # Set to True to enable debugging visuals

HEALTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
POSTURE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

REQUIRED_CONSECUTIVE_FRAMES = 14
CHANGE_THRESHOLD = 1.0

# Initialize buffers with deque to maintain a fixed length
health_update_buffer = {
    'player': deque(maxlen=REQUIRED_CONSECUTIVE_FRAMES),
    'boss': deque(maxlen=REQUIRED_CONSECUTIVE_FRAMES)
}
current_health = {
    'player': None,
    'boss': None
}

posture_update_buffer = {
    'player': deque(maxlen=REQUIRED_CONSECUTIVE_FRAMES),
    'boss': deque(maxlen=REQUIRED_CONSECUTIVE_FRAMES)
}
current_posture = {
    'player': None,
    'boss': None
}


@njit
def compute_health_percentage(w, total_width):
    """
    Compute the health percentage based on width and total width.

    Args:
        w (float): Width of the health bar.
        total_width (float): Total possible width of the health bar.

    Returns:
        float: Health percentage (0 to 100).
    """
    health_percentage = (w / total_width) * 100
    if health_percentage < 0:
        health_percentage = 0.0
    elif health_percentage > 100:
        health_percentage = 100.0
    return health_percentage


@njit
def compute_posture_percentage(indices, total_width):
    """
    Compute the posture percentage based on indices and maximum profile value.

    Args:
        indices (1D array): Indices where profile > threshold.
        total_width (float): Total possible width of the posture bar.

    Returns:
        float: Posture percentage (0 to 100).
    """
    if indices.size > 0:
        posture_bar_length = indices[-1] - indices[0]
        posture_percentage = (posture_bar_length / total_width) * 100
        if posture_percentage < 0:
            posture_percentage = 0.0
        elif posture_percentage > 100:
            posture_percentage = 100.0
    else:
        posture_percentage = 0.0
    return posture_percentage


def calculate_health_percentage(health_bar_image):
    """
    Calculate the health percentage based on the health bar length.

    Args:
        health_bar_image (np.ndarray): Image of the health bar.

    Returns:
        float: Health percentage (0 to 100).
    """
    hsv = cv2.cvtColor(health_bar_image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, HEALTH_KERNEL, iterations=2)

    # Apply adaptive threshold
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the health bar
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, _ = cv2.boundingRect(largest_contour)
        health_percentage = compute_health_percentage(w, binary_mask.shape[1])
    else:
        health_percentage = 0.0

    return health_percentage


def calculate_posture_percentage(posture_bar_image):
    """
    Calculate the posture percentage based on the posture bar length.

    Args:
        posture_bar_image (np.ndarray): Image of the posture bar.

    Returns:
        float: Posture percentage (0 to 100).
    """
    hsv = cv2.cvtColor(posture_bar_image, cv2.COLOR_BGR2HSV)

    # Define posture bar color range (yellow/orange)
    lower_color = np.array([10, 100, 100])
    upper_color = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, POSTURE_KERNEL, iterations=2)

    # Project the mask along the vertical axis to get a horizontal profile
    profile = np.sum(mask, axis=0)

    # Use Numba-accelerated function to compute posture percentage
    indices = np.where(profile > (np.max(profile) * 0.3))[0]
    posture_percentage = compute_posture_percentage(indices, mask.shape[1])

    return posture_percentage


def extract_health(player_health_img, boss_health_img):
    """
    Extract health percentages for the player and the boss.

    Args:
        player_health_img (np.ndarray): Image containing the player's health bar.
        boss_health_img (np.ndarray): Image containing the boss's health bar.

    Returns:
        tuple: (player_health, boss_health)
    """
    player_health = calculate_health_percentage(player_health_img)
    boss_health = calculate_health_percentage(boss_health_img)

    if DEBUG_MODE:
        cv2.imshow('Player Health Bar', player_health_img)
        cv2.imshow('Boss Health Bar', boss_health_img)
        cv2.waitKey(1)

    return player_health, boss_health


def extract_posture(player_posture_img, boss_posture_img):
    """
    Extract posture percentages for the player and the boss.

    Args:
        player_posture_img (np.ndarray): Image containing the player's posture bar.
        boss_posture_img (np.ndarray): Image containing the boss's posture bar.

    Returns:
        tuple: (player_posture, boss_posture)
    """
    player_posture = calculate_posture_percentage(player_posture_img)
    boss_posture = calculate_posture_percentage(boss_posture_img)

    if DEBUG_MODE:
        cv2.imshow('Player Posture Bar', player_posture_img)
        cv2.imshow('Boss Posture Bar', boss_posture_img)
        cv2.waitKey(1)

    return player_posture, boss_posture


def update_health(new_player_health, new_boss_health):
    """
    Update health values only when changes are detected over multiple consecutive frames.
    If the current health is 0 and a new non-zero value is detected, update immediately
    to handle player resurrection scenarios.

    Args:
        new_player_health: Newly detected player health value.
        new_boss_health: Newly detected boss health value.

    Returns:
        tuple: (current_player_health, current_boss_health)
    """
    global health_update_buffer, current_health

    def is_valid_update(current, new, max_change):
        return abs(new - current) <= max_change

    # Update Player Health
    if current_health['player'] is None:
        current_health['player'] = new_player_health
        logging.info(f"Initialized player health: {new_player_health:.2f}")
    else:
        if current_health['player'] <= 1.0 and new_player_health > 0.0:
            # Immediate update from 0 to non-zero
            logging.info(f"Player health immediately updated from 0.0 to {new_player_health:.2f}")
            current_health['player'] = new_player_health
            health_update_buffer['player'].clear()
        elif new_player_health <= 1.0 and current_health['player'] != 0.0:
            # Immediate update to 0
            logging.info(f"Player health immediately updated to 0.0")
            current_health['player'] = 0.0
            health_update_buffer['player'].clear()
        else:
            # Check if health has changed beyond the threshold
            if abs(new_player_health - current_health['player']) > CHANGE_THRESHOLD:
                health_update_buffer['player'].append(1)
            else:
                health_update_buffer['player'].append(0)

            # If change detected over required consecutive frames, update health
            if sum(health_update_buffer['player']) >= REQUIRED_CONSECUTIVE_FRAMES:
                logging.info(f"Player health updated: {current_health['player']:.2f} -> {new_player_health:.2f}")
                current_health['player'] = new_player_health
                health_update_buffer['player'].clear()

    # Update Boss Health
    if current_health['boss'] is None:
        current_health['boss'] = new_boss_health
        logging.info(f"Initialized boss health: {new_boss_health:.2f}")
    else:
        if not is_valid_update(current_health['boss'], new_boss_health, 20):
            current_health['boss'] = current_health['boss']
        if current_health['boss'] == 0.0 and new_boss_health > 0.0:
            # Immediate update from 0 to non-zero
            logging.info(f"Boss health immediately updated from 0.0 to {new_boss_health:.2f}")
            current_health['boss'] = new_boss_health
            health_update_buffer['boss'].clear()
        else:
            # Check if health has changed beyond the threshold
            if abs(new_boss_health - current_health['boss']) > CHANGE_THRESHOLD:
                health_update_buffer['boss'].append(1)
            else:
                health_update_buffer['boss'].append(0)

            # If change detected over required consecutive frames, update health
            if sum(health_update_buffer['boss']) >= REQUIRED_CONSECUTIVE_FRAMES:
                logging.info(f"Boss health updated: {current_health['boss']:.2f} -> {new_boss_health:.2f}")
                current_health['boss'] = new_boss_health
                health_update_buffer['boss'].clear()

    return current_health['player'], current_health['boss']


def update_posture(new_player_posture, new_boss_posture):
    """
    Update posture values only when changes are detected over multiple consecutive frames.
    If the current posture is 0 and a new non-zero value is detected, update immediately
    to handle player resurrection scenarios.

    Args:
        new_player_posture: Newly detected player posture value.
        new_boss_posture: Newly detected boss posture value.

    Returns:
        tuple: (current_player_posture, current_boss_posture)
    """
    global posture_update_buffer, current_posture

    def is_valid_update(current, new, max_change):
        return abs(new - current) <= max_change

    # Update Player Posture
    if current_posture['player'] is None:
        current_posture['player'] = new_player_posture
        logging.info(f"Initialized player posture: {new_player_posture:.2f}")
    else:
        if not is_valid_update(current_posture['player'], new_player_posture, 25):
            current_posture['player'] = current_posture['player']
        if current_posture['player'] == 0.0 and new_player_posture > 0.0:
            # Immediate update from 0 to non-zero
            logging.info(f"Player posture immediately updated from 0.0 to {new_player_posture:.2f}")
            current_posture['player'] = new_player_posture
            posture_update_buffer['player'].clear()
        else:
            # Check if posture has changed beyond the threshold
            if abs(new_player_posture - current_posture['player']) > CHANGE_THRESHOLD:
                posture_update_buffer['player'].append(1)
            else:
                posture_update_buffer['player'].append(0)

            # If change detected over required consecutive frames, update posture
            if sum(posture_update_buffer['player']) >= REQUIRED_CONSECUTIVE_FRAMES:
                logging.info(f"Player posture updated: {current_posture['player']:.2f} -> {new_player_posture:.2f}")
                current_posture['player'] = new_player_posture
                posture_update_buffer['player'].clear()

    # Update Boss Posture
    if current_posture['boss'] is None:
        current_posture['boss'] = new_boss_posture
    else:
        if not is_valid_update(current_posture['boss'], new_boss_posture, 20):
            current_posture['boss'] = current_posture['boss']
        if current_posture['boss'] == 0.0 and new_boss_posture > 0.0:
            # Immediate update from 0 to non-zero
            current_posture['boss'] = new_boss_posture
            posture_update_buffer['boss'].clear()
        else:
            # Check if posture has changed beyond the threshold
            if abs(new_boss_posture - current_posture['boss']) > CHANGE_THRESHOLD:
                posture_update_buffer['boss'].append(1)
            else:
                posture_update_buffer['boss'].append(0)

            # If change detected over required consecutive frames, update posture
            if sum(posture_update_buffer['boss']) >= REQUIRED_CONSECUTIVE_FRAMES:
                current_posture['boss'] = new_boss_posture
                posture_update_buffer['boss'].clear()

    return current_posture['player'], current_posture['boss']
