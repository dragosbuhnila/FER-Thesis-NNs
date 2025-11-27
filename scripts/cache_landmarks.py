import os; import sys
import time

import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import h5py
from joblib import Parallel, delayed

from modules.misc import hash_image, print_npy
from modules.config import ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, LANDMARK_COORDINATES_FOLDER_PATH, GLOBALS, RESULTS_LIGHT_PATH, MODULES_DIR
from modules.landmark_utils import detect_facial_landmarks



                                                                                                    # skip it cause it's too heavy, like 100MB the npy and 250MB the txt
def save_debugging_data(dumping_time, indices_to_hashes, landmark_coords_dict, conflicting_hashes, skip_landmark_coords_dict=True):
    """Save debugging data to disk."""
    # Save indices to hashes
    dump_file_indices = os.path.join(RESULTS_LIGHT_PATH, "debugging_hashes", f"{dumping_time}__indices_to_hashes.npy")
    dump_file_indices_readable = dump_file_indices.replace(".npy", "_readable.txt")
    os.makedirs(os.path.dirname(dump_file_indices), exist_ok=True)
    try:
        np.save(dump_file_indices, indices_to_hashes)
        print_npy(dump_file_indices, dump_file_indices_readable)
        print(f"Saved indices_to_hashes to {dump_file_indices} for debugging.")
    except Exception as e:
        print(f"Error saving indices_to_hashes: {e}")

    if skip_landmark_coords_dict:
        # Save landmark coordinates dictionary
        dump_file_landmark_dict = os.path.join(RESULTS_LIGHT_PATH, "debugging_hashes", f"{dumping_time}__landmark_coords_dict.npy")
        dump_file_landmark_dict_readable = dump_file_landmark_dict.replace(".npy", "_readable.txt")
        try:
            np.save(dump_file_landmark_dict, landmark_coords_dict)
            print_npy(dump_file_landmark_dict, dump_file_landmark_dict_readable)
            print(f"Saved landmark_coords_dict to {dump_file_landmark_dict} for debugging.")
        except Exception as e:
            print(f"Error saving landmark_coords_dict: {e}")

    # Save conflicting hashes
    dump_file_conflicting_hashes = os.path.join(RESULTS_LIGHT_PATH, "debugging_hashes", f"{dumping_time}__conflicting_hashes.npy")
    dump_file_conflicting_hashes_readable = dump_file_conflicting_hashes.replace(".npy", "_readable.txt")
    try:
        np.save(dump_file_conflicting_hashes, list(conflicting_hashes))
        print_npy(dump_file_conflicting_hashes, dump_file_conflicting_hashes_readable)
        print(f"Saved conflicting_hashes to {dump_file_conflicting_hashes} for debugging.")
    except Exception as e:
        print(f"Error saving conflicting_hashes: {e}")

    return dump_file_conflicting_hashes_readable


def append_hash_image_definition(dump_file_conflicting_hashes):
    """Append the definition of hash_image to the conflicting hashes file."""
    misc_file_path = os.path.abspath(os.path.join(MODULES_DIR, "misc.py"))
    if os.path.exists(misc_file_path):
        with open(misc_file_path, "r") as misc_file:
            misc_code = misc_file.readlines()

        # Extract the hash_image function definition
        hash_image_def = []
        inside_function = False
        for line in misc_code:
            if line.strip().startswith("def hash_image"):
                inside_function = True
            if inside_function:
                hash_image_def.append(line)
            if inside_function and line.strip() == "":  # End of function (empty line after it)
                break

        if hash_image_def:
            with open(dump_file_conflicting_hashes, "ab") as dump_file:
                dump_file.write(b"\n# Definition of hash_image\n")
                dump_file.write("".join(hash_image_def).encode("utf-8"))
            print(f"Appended the definition of hash_image to {dump_file_conflicting_hashes}.")
        else:
            print("Could not find the definition of hash_image in misc.py.")
    else:
        print(f"Could not find misc.py at {misc_file_path}.")


def handle_debugging_and_append_function(indices_to_hashes, landmark_coords_dict, conflicting_hashes):
    """Handle debugging data saving and append hash_image definition."""
    dumping_time = time.strftime('%Y%m%d-%H%M%S')
    dump_file_conflicting_hashes_readable = save_debugging_data(dumping_time, indices_to_hashes, landmark_coords_dict, conflicting_hashes)
    append_hash_image_definition(dump_file_conflicting_hashes_readable)


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




if __name__ == "__main__":
    os.makedirs(LANDMARK_COORDINATES_FOLDER_PATH, exist_ok=True)

    for dataset_path in DATASETS:
        print("=========================================")
        print(f"Loading landmarks for dataset: {dataset_path}")

        landmark_coords_dict = dict()
        indices_to_hashes = dict()
        conflicting_hashes = set()
        duplicates_found = False

        # 1) open the h5 file
        with h5py.File(dataset_path, 'r') as h5_file:
            for key in KEYS_OF_INTEREST:
                if key in h5_file.keys():
                    images = h5_file[key][...]

                    # 2) load landmark coordinates
                    image_hashes = [hash_image(img) for img in images]
                    
                    landmark_coords = detect_landmarks_parallel(images, image_hashes)

                    if len(landmark_coords) != images.shape[0]:
                        raise ValueError(f"Number of landmark coordinates ({len(landmark_coords)}) does not match number of images ({images.shape[0]})")

                    print(f"Unlandmarkable images list ({len(GLOBALS['UNLANDMARKABLE_IMAGES_LIST'])} items): ")
                    print(GLOBALS["UNLANDMARKABLE_IMAGES_LIST"])

                    i = 0
                    for img_hash, coords in zip(image_hashes, landmark_coords):
                        if img_hash in landmark_coords_dict:
                            print(f"Conflict detected: img_hash {img_hash} already exists in the dictionary.")
                            # NOTE: it may not be a conflict but just a repetition of the same image in the dataset!
                            conflicting_hashes.add(img_hash)
                        else:
                            landmark_coords_dict[img_hash] = coords

                        indices_to_hashes[f"{key}_{i}"] = img_hash
                        i += 1

                    if len(landmark_coords_dict) != len(landmark_coords):
                        duplicates_found = True
                        print(f"Number of unique landmark coordinates ({len(landmark_coords_dict)}) does not match number of landmark coordinates ({len(landmark_coords)}): diff = {len(landmark_coords) - len(landmark_coords_dict)}")

        print(f"Total images processed: {len(indices_to_hashes)}")

        if duplicates_found:
            print(f"Conflicting hashes found: {len(conflicting_hashes)}")
            handle_debugging_and_append_function(indices_to_hashes, landmark_coords_dict, conflicting_hashes)

                        
        # 3) save the landmark coordinates to disk
        for img_hash, landmark_coords in tqdm(landmark_coords_dict.items(), total=len(landmark_coords_dict)):
            os.makedirs(LANDMARK_COORDINATES_FOLDER_PATH, exist_ok=True)
            landmark_file_path = os.path.join(LANDMARK_COORDINATES_FOLDER_PATH, f"{img_hash}.npy")
            np.save(landmark_file_path, landmark_coords)


        print(f"Loaded landmark coordinates for dataset: {dataset_path}")
        print("=========================================")
        