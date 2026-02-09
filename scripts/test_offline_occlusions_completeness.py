import os; import sys;
sys.path.append(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__), '..'))))  # Add the project root to the path

import h5py
from tqdm import tqdm

from modules.config import BOSPHORUS_DUPLICATE_IMAGES, BOSPHORUS_UNLANDMARKABLE_IMAGES, CONSOLE_OUTPUTS_PATH, CONSOLE_OUTPUTS_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, EMOTIONS
from modules.misc import get_timestamp, hash_image, Tee
from modules.landmark_utils import detect_facial_landmarks, get_landmark_coordinate_sets_by_emotion, load_landmark_coordinates



REDIRECT_OUTPUT = True    

LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__test_offline_occlusions_completeness.txt")

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    tee_instance = Tee(LOG_FILE)  # Create a single Tee instance
    sys.stdout = tee_instance
    sys.stderr = tee_instance  # Use the same instance for stderr



def load_original_dataset_hashes(original_dataset_path, split):
    """Load the original dataset and calculate hashes for each image."""
    original_hashes = {}
    images_loaded = {}
    with h5py.File(original_dataset_path, 'r') as f:
        # Process training data
        if split == "train":
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            for i in range(len(X_train)):
                image = X_train[i]
                emotion = y_train[i]
                image_hash = hash_image(image)
                original_hashes[image_hash] = emotion
                images_loaded[image_hash] = image

        # Process validation data
        if split == "val":
            X_val = f['X_val'][:]
            y_val = f['y_val'][:]
            for i in range(len(X_val)):
                image = X_val[i]
                emotion = y_val[i]
                image_hash = hash_image(image)
                original_hashes[image_hash] = emotion
                images_loaded[image_hash] = image

    return original_hashes, images_loaded

def load_occluded_dataset_metadata(occluded_dataset_path, split):
    """Load the occluded dataset and extract hashes, occlusion types, and metadata."""
    occluded_metadata = {}
    with h5py.File(occluded_dataset_path, 'r') as f:
        original_hash_train = f[f'original_hash_{split}'][:].astype(str)
        occ_train = f[f'occ_{split}'][:]
        pos_or_neg_train = f[f'pos_or_neg_{split}'][:]
        y_train = f[f'y_{split}'][:]

        for i, image_hash in tqdm(enumerate(original_hash_train), total=len(original_hash_train)):
            occlusion_type = f"{'positive' if pos_or_neg_train[i] == 1 else 'negative'}_{EMOTIONS[occ_train[i]]}"
            occluded_metadata.setdefault(image_hash, []).append(occlusion_type)
    return occluded_metadata

def test_occlusion_completeness(original_hashes, occluded_metadata):
    """Test if all images in the original dataset have the expected occlusions."""
    missing_occlusions = {}
    for image_hash, emotion in original_hashes.items():
        if emotion == 4:  # Skip neutral emotion
            continue
        expected_occlusions = [
            f"positive_{EMOTIONS[i]}" for i in range(len(EMOTIONS)) if i != 4
        ] + [
            f"negative_{EMOTIONS[i]}" for i in range(len(EMOTIONS)) if i != 4 and i != emotion
        ]
        actual_occlusions = occluded_metadata.get(image_hash, [])
        for occlusion in expected_occlusions:
            if occlusion not in actual_occlusions:
                missing_occlusions.setdefault(image_hash, []).append((occlusion))
    return missing_occlusions

def test_landmarking(image_hash, image, occlusion_type, emotion):
    """Test if the landmarking process works for a given image and occlusion type."""
    # 1) First check if the landmark is cached
    cached_landmarks = load_landmark_coordinates(image_hash)
    if cached_landmarks is None:
        print(f"\t\t[ERROR] No cached landmarks found for image hash: {image_hash}, GT: {emotion}")
        return False

    # 2) If not cached, try to landmark the image with the specified occlusion
    landmarks = detect_facial_landmarks(image, image_hash, False, True, True)
    if landmarks is None or len(landmarks) == 0:
        print(f"\t\t[ERROR] Landmarking failed for image hash: {image_hash}, GT: {emotion}")
        return False
    landmark_sets = get_landmark_coordinate_sets_by_emotion(landmarks, occlusion_type.split('_')[1])
    if landmark_sets is None or len(landmark_sets) == 0 or len(landmark_sets[0]) == 0:
        print(f"\t\t[ERROR] Landmarking failed for image hash: {image_hash} with occlusion: {occlusion_type} (no valid landmark sets), GT: {emotion}")
        return False

if __name__ == "__main__":
    for split in ["train", "val"]:
        print(f"\n=== Testing occlusion completeness for {split} split =============================================================")
        print(f"Loading original dataset (split: {split})...")
        original_hashes, images_loaded = load_original_dataset_hashes(ORIGINAL_TRAIN_VAL_SET_H5_PATH, split)
        print(f"Loaded {len(original_hashes)} images from the original dataset.")

        print(f"Loading occluded dataset (split: {split})...")
        occluded_metadata = load_occluded_dataset_metadata(OCCLUDED_TRAIN_VAL_SET_H5_PATH, split)
        print(f"Loaded metadata for {len(occluded_metadata)} images from the occluded dataset.")

        print("Testing occlusion completeness...")
        missing_occlusions = test_occlusion_completeness(original_hashes, occluded_metadata)

        if missing_occlusions:
            print(f"Found {len(missing_occlusions)} missing occlusions:")
            for i, (image_hash, occlusions) in enumerate(missing_occlusions.items()):
                if image_hash in BOSPHORUS_UNLANDMARKABLE_IMAGES:
                    print(f"{i+1:2d}) Image hash: {image_hash} (GT emotion: {EMOTIONS[original_hashes[image_hash]]}) - UNLANDMARKABLE (skipping occlusion check)")
                    continue
                if image_hash in BOSPHORUS_DUPLICATE_IMAGES:
                    print(f"{i+1:2d}) Image hash: {image_hash} (GT emotion: {EMOTIONS[original_hashes[image_hash]]}) - DUPLICATE (skipping occlusion check)")
                    continue
                if not test_landmarking(image_hash, images_loaded[image_hash], occlusions[0], EMOTIONS[original_hashes[image_hash]]):
                    continue

                for occlusion in occlusions:
                    if not test_landmarking(image_hash, images_loaded[image_hash], occlusion, EMOTIONS[original_hashes[image_hash]]):
                        continue
                    print(f"   - Missing occlusion: {occlusion}")
        else:
            print("All images have the expected occlusions.")
        print("==================================================================================================================")
