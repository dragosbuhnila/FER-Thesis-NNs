import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from modules.config import CONSOLE_OUTPUTS_PATH



REDIRECT_OUTPUT = False
LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__hello_world__console_output.txt")

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout

if __name__ == "__main__":
    # print("hello_world")

    # from modules.config import LANDMARK_COORDINATES_FOLDER_PATH, LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE
    # landmarks_cache_size = len(os.listdir(LANDMARK_COORDINATES_FOLDER_PATH)) if os.path.exists(LANDMARK_COORDINATES_FOLDER_PATH) else 0
    # if landmarks_cache_size == LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE:
    #     print("Landmark coordinates cache is complete. Size is: ", landmarks_cache_size)
    # else:
    #     print(f"Landmark coordinates cache is incomplete. Size is: {landmarks_cache_size}, expected: {LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE}")

    exit(0)
