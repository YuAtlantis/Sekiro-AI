# main.py

import cv2
import logging
from screen_capture import grab_full_screen, grab_region
from health_posture import extract_health, extract_posture
from ocr_utils import get_remaining_uses

logging.basicConfig(level=logging.INFO)

DEBUG_MODE = False  # Set to True to enable debugging visuals


def main():
    """
    Main function to run the screen monitoring application.
    """
    regions = {
        'self_blood': (54, 562, 400, 576),
        'boss_blood': (57, 93, 288, 105),
        'self_posture': (395, 535, 635, 552),
        'boss_posture': (315, 73, 710, 88),
        'remaining_uses': (955, 570, 970, 588)
    }

    current_remaining_uses = 19

    while True:
        try:
            # Capture the full screen
            full_screen_img = grab_full_screen()

            # Extract all necessary regions
            screens = {name: grab_region(full_screen_img, region) for name, region in regions.items()}

            # Extract health percentages for the player and the boss
            self_health, boss_health = extract_health(
                screens['self_blood'], screens['boss_blood'])

            # Extract posture percentages for the player and the boss
            self_posture, boss_posture = extract_posture(
                screens['self_posture'], screens['boss_posture'])

            # Get remaining uses using OCR
            current_remaining_uses = get_remaining_uses(
                screens['remaining_uses'], current_remaining_uses)

            if DEBUG_MODE:
                # Additional debugging visuals can be handled within modules
                pass

            # Exit condition: press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info('Exit signal received. Exiting...')
                break

        except KeyboardInterrupt:
            logging.info('Program interrupted by user. Exiting...')
            break
        except Exception as e:
            logging.error(f'An error occurred: {e}')

    # Clean up OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
