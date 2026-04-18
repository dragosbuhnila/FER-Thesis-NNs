import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import shutil
from PIL import Image
import h5py
import numpy as np

from modules.misc import Tee

from modules.config import (
    CONSOLE_OUTPUTS_PATH,
    EMOTIONS,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_RESIZED_PATH,
    OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH,

    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_SAD_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_SURPRISE_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_SAD_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_SURPRISE_PATH,

    OCCLUDED_TEST_SET_RESIZED_POSITIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_SAD_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_SURPRISE_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SAD_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SURPRISE_PATH,

    OCCLUDED_TEST_SET_H5_POSITIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_SAD_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_SURPRISE_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_SAD_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_SURPRISE_PATH,
)



BASE_EMOTIONS = [emotion for emotion in EMOTIONS if emotion != "NEUTRAL"]
POSITIVITY = ["POSITIVE", "NEGATIVE"]

LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__split_occluded_dataset_in_occlusion_types.log")
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Tee(LOG_FILE_PATH)
sys.stderr = Tee(LOG_FILE_PATH) 



def parse_occlusion_type(filename):
    """Extract positivity and emotion from the filename."""
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename does not contain enough fields: {filename}")
    
    occlusion_field = parts[-2]  # The field before "match" or "mismatch"
    positivity, emotion = occlusion_field.split("-")[1], occlusion_field.split("-")[2].upper()
    positivity = positivity.upper()
    emotion = emotion.upper()

    if positivity not in POSITIVITY or emotion not in BASE_EMOTIONS:
        raise ValueError(f"Invalid positivity or emotion in filename: {filename}")
    
    return positivity, emotion

def split_images_by_occlusion_type(source_dir, output_dirs, create_folder_for_all_emotions=False):
    """Split images into directories based on occlusion type."""
    if create_folder_for_all_emotions:
        for _, subset_path in output_dirs.items():
            for emotion in EMOTIONS:
                emotion_folder = os.path.join(subset_path, emotion)
                os.makedirs(emotion_folder, exist_ok=True) 

    # for root, _, files in os.walk(source_dir):
    #     for file in files:
    #         if file.endswith(".png"):
    #             try:
    #                 positivity, emotion = parse_occlusion_type(file)
    #                 destination_dir = output_dirs[f"{positivity}_{emotion}"]
                    
    #                 # Maintain folder structure
    #                 relative_path = os.path.relpath(root, source_dir)
    #                 destination_folder = os.path.join(destination_dir, relative_path)
    #                 os.makedirs(destination_folder, exist_ok=True)

    #                 # Copy the file
    #                 source_path = os.path.join(root, file)
    #                 destination_path = os.path.join(destination_folder, file)
    #                 shutil.copy(source_path, destination_path)
    #                 print(f"Copied: {source_path} -> {destination_path}")
    #             except ValueError as e:
    #                 print(f"Skipping file due to error: {e}")

def load_h5_dataset(h5_path):
    """Load the dataset from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        X_test = np.array(f["X_test"])
        y_test = np.array(f["y_test"])
        class_names = np.array(f["class_names"])
        paths = np.array(f["paths"])
    return X_test, y_test, class_names, paths

def save_h5_dataset(h5_path, X, y, class_names, paths):
    """Save the dataset to an H5 file."""
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_test", data=X)
        f.create_dataset("y_test", data=y)
        f.create_dataset("class_names", data=class_names)
        f.create_dataset("paths", data=paths)

def split_h5_by_occlusion_type(h5_path, output_paths):
    """Split H5 dataset into files based on occlusion type."""
    X_test, y_test, class_names, paths = load_h5_dataset(h5_path)

    # Initialize dictionaries for each occlusion type
    datasets = {key: ([], [], []) for key in output_paths.keys()}

    for i, path in enumerate(paths):
        try:
            path_str = path.decode("utf-8")
            filename = os.path.basename(path_str)
            positivity, emotion = parse_occlusion_type(filename)
            key = f"{positivity}_{emotion}"

            datasets[key][0].append(X_test[i])
            datasets[key][1].append(y_test[i])
            datasets[key][2].append(path)
        except ValueError as e:
            print(f"Skipping entry due to error: {e}")

    # Save each dataset
    for key, (X, y, paths) in datasets.items():
        save_h5_dataset(output_paths[key], np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32), class_names, np.array(paths, dtype="S"))
        print(f"Dataset saved to: {output_paths[key]}")

    # Print summary of each dataset
    for key, (X, y, paths) in datasets.items():
        print(f"{key}: {len(X)} samples")

if __name__ == "__main__":
    # # Images dataset - unoccluded back resized 
    # unoccluded_back_dirs = {
    #     f"{pos}_{emo}": globals()[f"OCCLUDED_TEST_SET_UNOCCLUDED_BACK_{pos.upper()}_{emo}_PATH"]
    #     for pos in POSITIVITY for emo in BASE_EMOTIONS
    # }
    # split_images_by_occlusion_type(OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH, unoccluded_back_dirs)

    # Images dataset - resized
    resized_dirs = {
        f"{pos}_{emo}": globals()[f"OCCLUDED_TEST_SET_RESIZED_{pos.upper()}_{emo}_PATH"]
        for pos in POSITIVITY for emo in BASE_EMOTIONS
    }
    split_images_by_occlusion_type(OCCLUDED_TEST_SET_RESIZED_PATH, resized_dirs, create_folder_for_all_emotions=True)

    # # H5 dataset
    # h5_output_paths = {
    #     f"{pos}_{emo}": globals()[f"OCCLUDED_TEST_SET_H5_{pos.upper()}_{emo}_PATH"]
    #     for pos in POSITIVITY for emo in BASE_EMOTIONS
    # }
    # split_h5_by_occlusion_type(OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH, h5_output_paths)