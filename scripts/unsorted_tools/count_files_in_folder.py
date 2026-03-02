import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.config import OCCLUDED_TRAIN_SET_IMAGES_PATH, OCCLUDED_VAL_SET_IMAGES_PATH,\
                            OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,\
                            ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH

TRAIN_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH
VAL_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH


FOLDERS = {
    # "occluded_training_images": OCCLUDED_TRAIN_SET_IMAGES_PATH,
    # "occluded_validation_images": OCCLUDED_VAL_SET_IMAGES_PATH,
    "occluded_and_original_training_images": OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH,
    "occluded_and_original_validation_images": OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,
    "original_training_images": ORIGINAL_TRAIN_SET_IMAGES_PATH,
    "original_validation_images": ORIGINAL_VAL_SET_IMAGES_PATH,
}


total_counts = {}
for folder_name, folder_path in FOLDERS.items():
    print(f"Counting images in {folder_name}...")
    print(f"Folder path: {folder_path}")

    count = sum(len(files) for _, _, files in os.walk(folder_path))
    total_counts[folder_name] = count
    print(f"Total images in {folder_name}: {count}")
