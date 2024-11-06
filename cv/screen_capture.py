# screen_capture.py

import mss
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Initialize variables to calculate frame rate
frame_count = 0
start_time = time.time()


def grab_full_screen(region=(0, 0, 1024, 620)):
    """
    Capture a full screen or a specific region using mss.

    Args:
        region (tuple): A tuple defining the region to capture (left, top, right, bottom).

    Returns:
        np.ndarray: The captured image in BGR format.
    """
    global frame_count, start_time
    with mss.mss() as sct:
        monitor = {
            "top": region[1],
            "left": region[0],
            "width": region[2] - region[0],
            "height": region[3] - region[1]
        }
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        # Update frame count
        frame_count += 1

        # Calculate and log frame rate every 50 frames
        if frame_count % 50 == 0:
            elapsed_time = time.time() - start_time
            frame_rate = frame_count / elapsed_time
            logging.info(f"Current frame rate: {frame_rate:.2f} FPS")
            # Reset counters
            frame_count = 0
            start_time = time.time()

        return img[:, :, :3]


def grab_region(full_screen, region):
    """
    Extract a specific region from the full screen image.

    Args:
        full_screen (np.ndarray): The full screen image.
        region (tuple): A tuple defining the region to extract (left, top, right, bottom).

    Returns:
        np.ndarray: The extracted region image.
    """
    x1, y1, x2, y2 = region
    return full_screen[y1:y2, x1:x2]
