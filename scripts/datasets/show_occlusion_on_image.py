# This code is the same as show_trainset_dupes.py, but feel free to edit it to show specific images by index

import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import h5py
import matplotlib.pyplot as plt

from modules.config import EMOTIONS, ORIGINAL_TRAIN_VAL_SET_H5_PATH, ADELE_TEST_SET_H5_PATH
from modules.landmark_utils import detect_facial_landmarks, get_landmark_coordinate_sets_by_emotion, load_landmark_coordinates
from modules.mask import apply_mask_to_all_sets
from modules.misc import hash_image


# ORIGINAL_TRAIN_VAL_SET_H5_PATH or ADELE_TEST_SET_H5_PATH
DATASET_PATHS = {
    "test": ADELE_TEST_SET_H5_PATH,
    "trainval": ORIGINAL_TRAIN_VAL_SET_H5_PATH,
}

# X_train_10187

IMAGES_TO_SHOW = [
    {
        "img_key": "X_train", 
        "gt_key": "y_train",
        "index_to_show": 660,
    },  
]


if __name__ == "__main__":
    datasets = dict()
    for ds_name, ds_path in DATASET_PATHS.items():
        with h5py.File(ds_path, 'r') as h5_file:
            # Save the entire content of the HDF5 file
            datasets[ds_name] = {key: h5_file[key][...] for key in h5_file.keys()}

    for image_info in IMAGES_TO_SHOW:
        img_key = image_info["img_key"]
        gt_key = image_info["gt_key"]
        idx = image_info["index_to_show"]

        if "train" in img_key or "val" in img_key:
            ds_name = "trainval"
        elif "test" in img_key:
            ds_name = "test"
        else:
            raise ValueError(f"Unknown dataset for image key: {img_key}")

        img = datasets[ds_name][img_key][idx]
        gt = datasets[ds_name][gt_key][idx]
        emotion = EMOTIONS[gt]

        img_hash = hash_image(img)
        landmarks_detected = detect_facial_landmarks(img, img_hash, ignore_error=False)
        landmarks_cached = load_landmark_coordinates(img_hash)
        landmark_sets_detected = get_landmark_coordinate_sets_by_emotion(landmarks_detected, emotion)
        landmark_sets_cached = get_landmark_coordinate_sets_by_emotion(landmarks_cached, emotion)
        occluded_img_detected = apply_mask_to_all_sets(img, landmark_sets_detected, "lines")
        occluded_img_cached = apply_mask_to_all_sets(img, landmark_sets_cached, "lines")

        plt.figure(figsize=(8, 4))
        plt.suptitle(f'Image at index {img_key}_{idx}, {emotion} (i.e. {gt})\nHash: {img_hash}\nLeft: detected landmarks, Right: cached landmarks', fontsize=12)
        plt.subplot(1, 2, 1)
        plt.imshow(occluded_img_detected)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(occluded_img_cached)
        plt.axis('off')

        plt.show()
