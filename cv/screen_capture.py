# screen_capture.py

import mss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def grab_full_screen(region=(0, 0, 1024, 620)):
    """
    Capture a full screen or a specific region using mss.

    Args:
        region (tuple): A tuple defining the region to capture (left, top, right, bottom).

    Returns:
        np.ndarray: The captured image in BGR format.
    """
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
