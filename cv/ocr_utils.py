# ocr_utils.py

import cv2
import pytesseract
import logging

logging.basicConfig(level=logging.INFO)

DEBUG_MODE = False  # Set to True to enable debugging visuals


def get_remaining_uses(screenshot, current_remaining):
    """
    Extract the remaining uses of items using OCR.

    Args:
        screenshot (np.ndarray): Image containing the remaining uses text.
        current_remaining (int): Current count of remaining uses.

    Returns:
        int: Updated count of remaining uses.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Binarize the image using Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR configuration to whitelist digits
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    extracted_text = pytesseract.image_to_string(binary, config=config, lang='eng').strip()

    try:
        remaining_uses = int(extracted_text)
        current_remaining = remaining_uses
        logging.info(f'Remaining Uses: {current_remaining}')
    except ValueError:
        logging.warning(f'OCR failed to parse remaining uses: "{extracted_text}"')

    if DEBUG_MODE:
        cv2.imshow('Original Image', screenshot)
        cv2.imshow('Processed Image', binary)

    return current_remaining
