import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import shutil
from PIL import Image

from modules.config import OCCLUDED_TEST_SET_RESIZED_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_IMAGES_PATH, BOSPHORUS_TEST_HQ_IMAGES_PATH, \
                            IMAGES_SHAPE


# Constants
IMAGES_SOURCE_FOLDER_PATH = OCCLUDED_TEST_SET_RESIZED_PATH
IMAGES_DESTINATION_FOLDER_PATH = OCCLUDED_TEST_SET_UNOCCLUDED_BACK_IMAGES_PATH
UNOCCLUDED_IMAGES_FOLDER_PATH = BOSPHORUS_TEST_HQ_IMAGES_PATH
IMG_WIDTH = IMAGES_SHAPE[0]
IMG_HEIGHT = IMAGES_SHAPE[1]

def process_images():
    # Ensure the destination folder exists
    os.makedirs(IMAGES_DESTINATION_FOLDER_PATH, exist_ok=True)

    # Walk through the source folder
    for root, _, files in os.walk(IMAGES_SOURCE_FOLDER_PATH):
        for file in files:
            if file.endswith(".png"):  # Process only .png files
                relative_path = os.path.relpath(root, IMAGES_SOURCE_FOLDER_PATH)
                
                # Extract the base filename (before the double underscore __)
                base_name = file.split("__")[0]
                if not base_name.startswith("image"):
                    raise ValueError(f"the extracted name from a filename should start with image_idx but instead found: {file}")
                
                bosphorus_name = file.split("__")[1]
                if not bosphorus_name.startswith("bosphorus"):
                    raise ValueError(f"the extracted name from a filename's second part (after __) should start with bosphorus but instead found: {file}")

                # Construct the path to the unoccluded image
                unoccluded_image_path = os.path.join(UNOCCLUDED_IMAGES_FOLDER_PATH, relative_path, f"{bosphorus_name}.png")

                # Construct the destination path
                destination_folder = os.path.join(IMAGES_DESTINATION_FOLDER_PATH, relative_path)
                os.makedirs(destination_folder, exist_ok=True)
                destination_path = os.path.join(destination_folder, file)

                # Copy and resize the unoccluded image
                if os.path.exists(unoccluded_image_path):
                    try:
                        with Image.open(unoccluded_image_path) as img:
                            # Resize the image
                            resized_img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
                            # Save the resized image to the destination path
                            resized_img.save(destination_path)
                            print(f"Resized and saved: {unoccluded_image_path} -> {destination_path}")
                    except Exception as e:
                        print(f"Error processing {unoccluded_image_path}: {e}")
                else:
                    print(f"\t[ERROR] Unoccluded image not found for: {file}")
                    print(f"\t\t Tried path: {unoccluded_image_path}")

if __name__ == "__main__":
    process_images()