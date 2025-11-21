import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.landmark_utils import detect_facial_landmarks_frompath
from modules.config import GLOBALS, ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH, ADELE_TEST_SET_IMAGES_PATH



IMAGES_PATHS = [
    os.path.join(ORIGINAL_TRAIN_SET_IMAGES_PATH, "ANGRY", "7db37b671f4ea6fc06d5a33398c4042a.png"),
]



if __name__ == "__main__":
    for image_path in IMAGES_PATHS:
        landmarks = detect_facial_landmarks_frompath(image_path)
        print(f"Image: {image_path}")
        print(f"Detected landmarks: {landmarks}")

        total_landmarks_loaded = GLOBALS["TOTAL_IMAGES_LANDMARKS_LOADED"]
        total_landmarks_saved = GLOBALS["TOTAL_IMAGES_LANDMARKS_SAVED"]
        print(f"Total landmarks loaded: {total_landmarks_loaded}")
        print(f"Total landmarks saved: {total_landmarks_saved}")

