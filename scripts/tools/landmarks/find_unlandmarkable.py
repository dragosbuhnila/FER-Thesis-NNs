import os; import sys
import time

import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import h5py
from joblib import Parallel, delayed

from modules.misc import hash_image
from modules.config import ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, LANDMARK_COORDINATES_FOLDER_PATH, GLOBALS, RESULTS_LIGHT_PATH, MODULES_DIR
from modules.landmark_utils import detect_facial_landmarks



def detect_landmarks_parallel(images, image_hashes, n_jobs=-1):
    """Detect landmarks in parallel using joblib."""
    def detect_single(img, img_hash):
        return detect_facial_landmarks(img, img_hash, ignore_error=True, force_recalculate=True, dont_save=True)

    # Use joblib.Parallel to parallelize the detection with tqdm for progress tracking
    landmark_coords = Parallel(n_jobs=n_jobs)(
        delayed(detect_single)(img, img_hash) for img, img_hash in tqdm(zip(images, image_hashes), total=len(images), desc="Detecting landmarks")
    )
    return landmark_coords



DATASETS = [
    ADELE_TEST_SET_H5_PATH,       
        # Verifying new h5 contents:
        # X_test.shape: (350, 128, 128, 3)
        # X_test dtype: uint8
        # class_names.shape: (7,)
        # class_names: [b'ANGRY' b'DISGUST' b'FEAR' b'HAPPY' b'NEUTRAL' b'SAD' b'SURPRISE']
        # y_test.shape: (350,)
        # y_test: [0 0 0 ... 6 6 6]      
    ORIGINAL_TRAIN_VAL_SET_H5_PATH,
        # Verifying new h5 contents:
        # X_train.shape: (21332, 128, 128, 3)
        # X_train dtype: uint8
        # X_val.shape: (5273, 128, 128, 3)
        # X_val dtype: uint8
        # class_names.shape: (7,)
        # class_names: [b'ANGRY' b'DISGUST' b'FEAR' b'HAPPY' b'NEUTRAL' b'SAD' b'SURPRISE']
        # y_train.shape: (21332,)
        # y_train: [0 0 0 ... 6 6 6]
        # y_val.shape: (5273,)
        # y_val: [0 0 0 ... 6 6 6]
]

KEYS_OF_INTEREST = [
    "X_test",
    "X_train",
    "X_val",
]

UNLANDMARKABLE_HASHES = {}
for key in KEYS_OF_INTEREST:
    UNLANDMARKABLE_HASHES[key] = []



# In order to know the index of a certain image (by knowing its hash, which is the only output this script provides) you should run the caching script
if __name__ == "__main__":
    os.makedirs(LANDMARK_COORDINATES_FOLDER_PATH, exist_ok=True)

    for dataset_path in DATASETS:
        print("=========================================")
        print(f"Loading landmarks for dataset: {dataset_path}")

        hashes_to_indices = dict()

        # 1) open the h5 file
        with h5py.File(dataset_path, 'r') as h5_file:
            for key in KEYS_OF_INTEREST:
                if key in h5_file.keys():
                    images = h5_file[key][...]

                    # 2) load landmark coordinates
                    image_hashes = [hash_image(img) for img in images]

                    landmark_coords_batch = detect_landmarks_parallel(images, image_hashes)
                    
                    i = 0
                    for coords, img_hash in zip(landmark_coords_batch, image_hashes):
                        if len(coords) == 0:
                            UNLANDMARKABLE_HASHES[key].append(img_hash)
                        # Only store the first occurrence of the hash (although there's just 4 dupes and all in train, which has no unlandmarkables)
                        if hashes_to_indices.get(img_hash) is None:
                            hashes_to_indices[img_hash] = f"{key}_{i}"
                        i += 1

                    if len(landmark_coords_batch) != images.shape[0]:
                        raise ValueError(f"Number of landmarkings executed ({len(landmark_coords_batch)}) does not match number of images ({images.shape[0]})")

                    print(f"Finished processing key: {key} in dataset: {dataset_path}:")
                    print(f"\tNumber of unlandmarkable images in {key}: {len(UNLANDMARKABLE_HASHES[key])}")
            

    print("=========================================")
    total_unlandmarkable = 0
    for key in KEYS_OF_INTEREST:
        num_unlandmarkable = len(UNLANDMARKABLE_HASHES[key])
        total_unlandmarkable += num_unlandmarkable
        print(f"Number of unlandmarkable images in {key}: {num_unlandmarkable}")
    print(f"Total number of unlandmarkable images in dataset: {total_unlandmarkable}")
                    
    # 3) save the unlandmarkable images' hashes to a txt file
    unlandmarkable_txt_path = os.path.join(RESULTS_LIGHT_PATH, "debugging_hashes", f"{time.strftime('%Y%m%d-%H%M%S')}__unlandmarkable_images.txt")
    os.makedirs(os.path.dirname(unlandmarkable_txt_path), exist_ok=True)
    with open(unlandmarkable_txt_path, 'w') as f:
        for key in KEYS_OF_INTEREST:
            f.write(f"Unlandmarkable images in {key} ({len(UNLANDMARKABLE_HASHES[key])}):\n")
            for img_hash in UNLANDMARKABLE_HASHES[key]:
                # output should look like "20eb13a5aee7da2826e3f3db18a2ba70": {"split": "X_train", "idx_to_remove": 19},
                f.write(f'\t"{img_hash}": {{"split": "{key}", "idx_to_remove": {hashes_to_indices[img_hash].split("_")[-1]}}},\n')
            f.write("\n")

    print(f"Loaded landmark coordinates for dataset: {dataset_path}")
    print("=========================================")
        