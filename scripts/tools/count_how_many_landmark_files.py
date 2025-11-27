import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import h5py

from modules.config import LANDMARK_COORDINATES_FOLDER_PATH, \
                        ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, \
                        ADELE_TEST_SET_IMAGES_PATH, ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH

LANDMARKS_PATH = LANDMARK_COORDINATES_FOLDER_PATH

IMAGES_FOLDERS = [
    ORIGINAL_TRAIN_SET_IMAGES_PATH, 
    ORIGINAL_VAL_SET_IMAGES_PATH,
    ADELE_TEST_SET_IMAGES_PATH,
    ]

H5_FILES_PATHS = [
    ADELE_TEST_SET_H5_PATH,
    ORIGINAL_TRAIN_VAL_SET_H5_PATH,
]

if __name__ == "__main__":
    # 1) Count how many .npy files are in the LANDMARKS_PATH directory
    count = 0
    other_count = 0
    for filename in os.listdir(LANDMARKS_PATH):
        if filename.endswith(".npy"):
            count += 1
        else:
            other_count += 1
    print(f"Number of landmark files in '{LANDMARKS_PATH}': {count}")
    print(f" Other count is {other_count}")

    # 2) Print the dimensions of the datasets from each images folder
    print("=============================================================")
    total_images = 0
    for images_folder in IMAGES_FOLDERS:
        emotion_folders = [ os.path.join(images_folder, d) for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d)) ]
        image_count = 0
        for emotion_folder in emotion_folders:
            num_images = len([filename for filename in os.listdir(emotion_folder) if os.path.isfile(os.path.join(emotion_folder, filename))])
            image_count += num_images
        total_images += image_count
        print(f"Number of images in '{images_folder}': {image_count}")
    print(f"Total number of images across all folders: {total_images}")

    # 3) Print the dimensions of the datasets from the h5 files
    print("=============================================================")
    total = 0
    for h5_file_path in H5_FILES_PATHS:
        with h5py.File(h5_file_path, "r") as f:
            for key in f.keys():
                # print(f"Dataset '{key}' in file '{h5_file_path}' has shape: {f[key].shape}")
                if key in ["X_train", "X_val", "X_test"]:   
                    total += f[key].shape[0]
                    print(f"Number of images in dataset '{key}' from file '{h5_file_path}': {f[key].shape[0]}")
    total += 350
    print(f"Total number of entries in h5 files (train + val + test): {total}")