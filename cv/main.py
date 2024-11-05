import concurrent.futures
import logging
import cv2
import time
from screen_capture import grab_region, grab_full_screen
from health_posture import extract_health, extract_posture

logging.basicConfig(level=logging.INFO)


def main():
    regions = {
        'game_window': (220, 145, 800, 530),
        'self_blood': (54, 562, 400, 576),
        'boss_blood': (57, 93, 288, 105),
        'self_posture': (395, 535, 635, 552),
        'boss_posture': (315, 73, 710, 88)
    }

    log_interval = 2
    last_log_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            # Capture a single frame
            full_screen_img = grab_full_screen()

            # Extract images for each region from the same frame
            self_blood_img = grab_region(full_screen_img, regions['self_blood'])
            boss_blood_img = grab_region(full_screen_img, regions['boss_blood'])
            self_posture_img = grab_region(full_screen_img, regions['self_posture'])
            boss_posture_img = grab_region(full_screen_img, regions['boss_posture'])

            # Submit health and posture extraction tasks
            health_future = executor.submit(extract_health, self_blood_img, boss_blood_img)
            posture_future = executor.submit(extract_posture, self_posture_img, boss_posture_img)

            # Wait for both tasks to complete
            self_health, boss_health = health_future.result()
            self_posture, boss_posture = posture_future.result()

            # Check if the log interval has passed
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                # Log the results together in one line
                logging.info(f'Player Health: {self_health:.2f}%, Boss Health: {boss_health:.2f}%, '
                             f'Player Posture: {self_posture:.2f}%, Boss Posture: {boss_posture:.2f}%')

                # Update the last log time
                last_log_time = current_time

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info('Exit signal received. Exiting...')
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
