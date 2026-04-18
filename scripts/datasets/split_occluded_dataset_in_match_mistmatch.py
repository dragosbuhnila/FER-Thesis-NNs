import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Add the project root to the path

import shutil
from PIL import Image
import h5py
import numpy as np

from modules.config import (
    EMOTIONS,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MATCHING_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MISMATCHING_RESIZED_IMAGES_PATH,

    OCCLUDED_TEST_SET_RESIZED_PATH,
    OCCLUDED_TEST_SET_RESIZED_MATCHING_PATH,
    OCCLUDED_TEST_SET_RESIZED_MISMATCHING_PATH,

    OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH,
    OCCLUDED_TEST_SET_H5_MATCHING_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_H5_MISMATCHING_RESIZED_IMAGES_PATH,
)



def split_images_dataset(source_dir, matching_dir, mismatching_dir, create_folder_for_all_emotions=False):
    # Ensure output directories exist
    os.makedirs(matching_dir, exist_ok=True)
    os.makedirs(mismatching_dir, exist_ok=True)

    if create_folder_for_all_emotions:
        for subset_path in [matching_dir, mismatching_dir]:
            for emotion in EMOTIONS:
                emotion_folder = os.path.join(subset_path, emotion)
                os.makedirs(emotion_folder, exist_ok=True) 


    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".png"):  # Process only .png files
                # Determine if the file is a match or mismatch
                if "mismatch" in file:
                    destination_base = mismatching_dir
                elif "match" in file:
                    destination_base = matching_dir
                else:
                    print(f"Skipping file (no match/mismatch in name): {file}")
                    continue

                # Maintain folder structure
                relative_path = os.path.relpath(root, source_dir)
                destination_folder = os.path.join(destination_base, relative_path)
                os.makedirs(destination_folder, exist_ok=True)

                # Copy the file to the appropriate folder
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                shutil.copy(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")



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

def split_h5_dataset(h5_path, matching_h5_path, mismatching_h5_path):
    # Load the original dataset
    X_test, y_test, class_names, paths = load_h5_dataset(h5_path)

    # Initialize lists for matching and mismatching datasets
    X_matching, y_matching, paths_matching = [], [], []
    X_mismatching, y_mismatching, paths_mismatching = [], [], []

    # Iterate through the dataset and split based on filenames
    for i, path in enumerate(paths):
        path_str = path.decode("utf-8")  # Convert bytes to string
        if "mismatch" in path_str:  # Check for mismatching images first
            X_mismatching.append(X_test[i])
            y_mismatching.append(y_test[i])
            paths_mismatching.append(path)
        elif "match" in path_str:  # Check for matching images
            X_matching.append(X_test[i])
            y_matching.append(y_test[i])
            paths_matching.append(path)

    # Convert lists to numpy arrays
    X_matching = np.array(X_matching, dtype=np.uint8)
    y_matching = np.array(y_matching, dtype=np.int32)
    paths_matching = np.array(paths_matching, dtype="S")  # Save as bytes

    X_mismatching = np.array(X_mismatching, dtype=np.uint8)
    y_mismatching = np.array(y_mismatching, dtype=np.int32)
    paths_mismatching = np.array(paths_mismatching, dtype="S")  # Save as bytes

    # Save the new datasets
    save_h5_dataset(matching_h5_path, X_matching, y_matching, class_names, paths_matching)
    save_h5_dataset(mismatching_h5_path, X_mismatching, y_mismatching, class_names, paths_mismatching)

    print(f"Matching dataset saved to: {matching_h5_path}")
    print(f"Mismatching dataset saved to: {mismatching_h5_path}")



if __name__ == "__main__":
    # split_images_dataset(OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MATCHING_RESIZED_IMAGES_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MISMATCHING_RESIZED_IMAGES_PATH)

    split_images_dataset(OCCLUDED_TEST_SET_RESIZED_PATH, OCCLUDED_TEST_SET_RESIZED_MATCHING_PATH, OCCLUDED_TEST_SET_RESIZED_MISMATCHING_PATH, create_folder_for_all_emotions=True)

    # split_h5_dataset(OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH, OCCLUDED_TEST_SET_H5_MATCHING_RESIZED_IMAGES_PATH, OCCLUDED_TEST_SET_H5_MISMATCHING_RESIZED_IMAGES_PATH)