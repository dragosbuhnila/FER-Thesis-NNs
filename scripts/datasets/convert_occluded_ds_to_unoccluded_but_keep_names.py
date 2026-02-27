import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import shutil
from PIL import Image
import argparse

from modules.config import OCCLUDED_TEST_SET_RESIZED_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH, BOSPHORUS_TEST_HQ_IMAGES_PATH, \
                            IMAGES_SHAPE, \
                            OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_LARGE_IMAGES_PATH



def check_filename_start(filename, expected_start):
    if not filename.startswith(expected_start):
        raise ValueError(f"the extracted name from a filename should start with {expected_start} but instead found: {filename}")


def extract_bosphorus_name_from_filename_with_imageidx(filename):
    # Extract the base filename (before the double underscore __)
    check_filename_start(filename, "image_")

    bosphorus_name = filename.split("__")[1]
    if not bosphorus_name.startswith("bosphorus"):
        raise ValueError(f"the extracted name from a filename's second part (after __) should start with bosphorus but instead found: {filename}")
    
    return bosphorus_name

def extract_bosphorus_name_from_filename_wo_imageidx(filename):
    # Extract the base filename (before the double underscore __)
    check_filename_start(filename, "bosphorus_bs")

    bosphorus_name = filename.split("__")[0]
    if not bosphorus_name.startswith("bosphorus"):
        raise ValueError(f"the extracted name from a filename's first part (before __) should start with bosphorus but instead found: {filename}")
    
    return bosphorus_name

# ============================================== SETTINGS ==============================================

parser = argparse.ArgumentParser(description="Convert occluded dataset to unoccluded but keep names")
parser.add_argument("--set", type=str, choices=["occluded_small", "occluded_large"], default="occluded_large", help="Which occluded dataset to process")
args = parser.parse_args()

if args.set == "occluded_small":
    IMAGES_SOURCE_FOLDER_PATH = OCCLUDED_TEST_SET_RESIZED_PATH
    IMAGES_DESTINATION_FOLDER_PATH = OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH
    UNOCCLUDED_IMAGES_FOLDER_PATH = BOSPHORUS_TEST_HQ_IMAGES_PATH
    # use this function if filenames are like image_001__bosphorus_bs001_01.png
    EXTRACT_BOSPHORUS_NAME_FROM_FILENAME_FUNCTION = extract_bosphorus_name_from_filename_with_imageidx

    DO_RESIZE = True
    IMG_WIDTH = IMAGES_SHAPE[0]
    IMG_HEIGHT = IMAGES_SHAPE[1]
elif args.set == "occluded_large":
    IMAGES_SOURCE_FOLDER_PATH = OCCLUDED_TEST_SET_IMAGES_PATH
    IMAGES_DESTINATION_FOLDER_PATH = OCCLUDED_TEST_SET_UNOCCLUDED_BACK_LARGE_IMAGES_PATH
    UNOCCLUDED_IMAGES_FOLDER_PATH = BOSPHORUS_TEST_HQ_IMAGES_PATH
    # use this function if filenames are like bosphorus_bs001_01__bosphorus_bs001_01.png
    EXTRACT_BOSPHORUS_NAME_FROM_FILENAME_FUNCTION = extract_bosphorus_name_from_filename_wo_imageidx

    DO_RESIZE = False
    IMG_WIDTH = None
    IMG_HEIGHT = None

print("=================== SETTINGS ===================")
print("ARGS:")
print(f"\tset: {args.set}")
print("CONSTANTS:")
print(f"\tIMAGES_SOURCE_FOLDER_PATH: {IMAGES_SOURCE_FOLDER_PATH}")
print(f"\tIMAGES_DESTINATION_FOLDER_PATH: {IMAGES_DESTINATION_FOLDER_PATH}")
print(f"\tUNOCCLUDED_IMAGES_FOLDER_PATH: {UNOCCLUDED_IMAGES_FOLDER_PATH}")
print(f"\tDO_RESIZE: {DO_RESIZE}")
print(f"\tIMG_WIDTH: {IMG_WIDTH}")
print(f"\tIMG_HEIGHT: {IMG_HEIGHT}")
print("================================================")

# ==============================================================================================================


def process_images():
    # Ensure the destination folder exists
    os.makedirs(IMAGES_DESTINATION_FOLDER_PATH, exist_ok=True)

    # Walk through the source folder
    for root, _, files in os.walk(IMAGES_SOURCE_FOLDER_PATH):
        for file in files:
            if file.endswith(".png"):  # Process only .png files
                relative_path = os.path.relpath(root, IMAGES_SOURCE_FOLDER_PATH)
                
                # Extract the base filename (before the double underscore __)
                bosphorus_name = EXTRACT_BOSPHORUS_NAME_FROM_FILENAME_FUNCTION(file)

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
                            if DO_RESIZE:
                                img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
                                img.save(destination_path)
                                print(f"Resized and saved: {unoccluded_image_path} -> {destination_path}")
                            else:
                                img.save(destination_path)
                                print(f"Saved without resizing: {unoccluded_image_path} -> {destination_path}")
                    except Exception as e:
                        print(f"Error processing {unoccluded_image_path}: {e}")
                else:
                    print(f"\t[ERROR] Unoccluded image not found for: {file}")
                    print(f"\t\t Tried path: {unoccluded_image_path}")

if __name__ == "__main__":
    process_images()