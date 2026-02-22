import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.config import OCCLUDED_TRAIN_SET_IMAGES_PATH, OCCLUDED_VAL_SET_IMAGES_PATH


total_train_images = sum(len(files) for _, _, files in os.walk(OCCLUDED_TRAIN_SET_IMAGES_PATH))
total_val_images = sum(len(files) for _, _, files in os.walk(OCCLUDED_VAL_SET_IMAGES_PATH))

print(f"Total occluded training images: {total_train_images}")
print(f"Total occluded validation images: {total_val_images}")