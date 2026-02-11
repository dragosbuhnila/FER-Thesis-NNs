import os
import sys
from PIL import Image
import h5py
import numpy as np
import hashlib
import time

from modules.config import EMOTIONS, IMAGES_SHAPE



# =============================================================================================================================
# ============================ Time and Filenames =============================================================================
# ====== get_timestamp, extract_info_from_occludedtrainvalset_filename ======================================================== 
# ======== create_occludedtrainvalset_filename_from_info, Tee =================================================================
# =============================================================================================================================

def get_timestamp(format="date-time"):
    if format == "date-time":
        return time.strftime('%Y%m%d-%H%M%S')
    else:
        raise NotImplementedError(f"Timestamp format {format} is not implemented.")

def extract_info_from_occludedtrainvalset_filename(filename):
    # Example of filename 00cd28ba22c246af733c4e0d8c6da551_gt-angry_occ-fear_mismatching_negative.png

    # Remove png or jpg if present
    if filename.endswith('.png'):
        filename = filename[:-4]
    elif filename.endswith('.jpg'):
        filename = filename[:-4]
    
    # First of all check if the format is correct
    parts = filename.split('_')
    if len(parts) != 5:
        raise ValueError(f"Filename {filename} does not conform to expected format. Should have 5 parts separated by underscores.")
    
    hash, gt_emotion_long, occ_emotion_long, mismatching, pos_or_neg = parts

    gt_emotion = gt_emotion_long.split('-')[1]
    occ_emotion = occ_emotion_long.split('-')[1]

    return hash, gt_emotion, occ_emotion, mismatching, pos_or_neg

def create_occludedtrainvalset_filename_from_info(hash, gt_emotion, occ_emotion, mismatching, pos_or_neg):
    filename = f"{hash}_gt-{gt_emotion}_occ-{occ_emotion}_{mismatching}_{pos_or_neg}.png"
    return filename

class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # Ensure the file is updated immediately
        self.stdout.write(data)
        self.stdout.flush()  # Ensure the terminal displays the output immediately

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


# ====================================================================================================================================
# ============================ End of Time and Filenames =============================================================================
# ====================================================================================================================================


# ====================================================================================================================================
# ============================ Computing Functions ===================================================================================
# ================================ hash_image ======================================================================================== 
# ====================================================================================================================================

def hash_image(image):
    # If a PIL Image, convert to numpy in a deterministic way
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    arr = np.ascontiguousarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()

# ====================================================================================================================================
# ============================ End of Computing Functions ============================================================================



# ============================ Print and Visualization Functions =============================

def print_npy(npy_file_path, output_file_path):
    data = np.load(npy_file_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()  # Convert 0-dim array to its content

    # Print the contents
    with open(output_file_path, 'w') as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"Index: {key} -> Hash: {value}\n")
        elif isinstance(data, list):
            for index, value in enumerate(data):
                f.write(f"Index: {index} -> Value: {value}\n")
        else:
            f.write(f"Contents of {npy_file_path}:\n")
            f.write(data.__str__())

# ============================ End of Print and Visualization Functions =============================



# ============================ Highly Coupled Functions =============================

def generate_h5_from_images(test_set_path, resized_path, h5_path):
    class_names = sorted(os.listdir(test_set_path))
    paths = []
    X_test = []
    y_test = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(test_set_path, class_name)
        image_files = sorted(os.listdir(class_folder))
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            # Load PNG image as RGB NumPy array and resize to (128, 128, 3)
            image = np.array(Image.open(image_path).convert('RGB').resize((IMAGES_SHAPE[0], IMAGES_SHAPE[1])))
            X_test.append(image)
            y_test.append(class_idx)  # Store class index instead of name
            paths.append(image_path)

            # Also save the images to resized_path
            save_folder = os.path.join(resized_path, class_name)
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(save_folder, image_file))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 3) Save new h5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)  # Now integers
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))  # Save as bytes
        f.create_dataset("paths", data=np.array(paths).astype('S'))  # Save as bytes
    print(f"Saved {X_test.shape[0]} images to {h5_path}")

    # 4) Check saved h5 to have 350 images, 7 classes, 50 images per class
    with h5py.File(h5_path, "r") as f:
        if "X_test" not in f.keys():
            raise ValueError("X_test not found in the H5 file.")
        if "y_test" not in f.keys():
            raise ValueError("y_test not found in the H5 file.")
        if "class_names" not in f.keys():
            raise ValueError("class_names not found in the H5 file.")
        if "paths" not in f.keys():
            raise ValueError("paths not found in the H5 file.")
        
        X_test_loaded = np.array(f["X_test"])
        y_test_loaded = np.array(f["y_test"])
        class_names_loaded = [name.decode('utf-8') for name in f["class_names"][...]]
        paths_loaded = [path.decode('utf-8') for path in f["paths"][...]]

        if X_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 images, but found {X_test_loaded.shape[0]}.")
        if y_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 labels, but found {y_test_loaded.shape[0]}.")
        if len(class_names_loaded) != 7:
            raise ValueError(f"Expected 7 classes, but found {len(class_names_loaded)}.")
        if len(paths_loaded) != 350:
            raise ValueError(f"Expected 350 paths, but found {len(paths_loaded)}.")

class StatsTracker:
    def __init__(self, emotions, generator_name, specific_mismatch=None, positive_or_negative=None):
        self.generator_name = generator_name
        self.specific_mismatch = specific_mismatch
        self.positive_or_negative = positive_or_negative

        # Initialize the stats dictionary
        self.stats = {
            # Processing
            'processed_images': 0,
            'skipped_images': 0,
            'saved_images': 0,
            # Types of images
            'matching_images': 0,
            'mismatching_images': 0,
            'positive_images': 0,
            'negative_images': 0,
        }

        # Add emotion-specific stats
        for emotion in emotions:
            self.stats[f'gt-{emotion.lower()}_images'] = 0
            self.stats[f'occ-{emotion.lower()}_images'] = 0

    def __str__(self):
        stats_str =  "=== Stats Tracker ===\n"
        stats_str += f"Generator Name: {self.generator_name}\n"
        stats_str += f"Specific Mismatch: {self.specific_mismatch}\n" if self.specific_mismatch else ""
        stats_str += f"Positive or Negative: {self.positive_or_negative}\n" if self.positive_or_negative else ""
        for key, value in self.stats.items():
            stats_str += f"{key}: {value}\n"
        stats_str += "=====================\n"
        return stats_str

    def update_from_filename(self, filename):
        # Parse the filename
        _, gt_emotion, occ_emotion, mismatching, pos_or_neg = extract_info_from_occludedtrainvalset_filename(filename)

        # Update matching/mismatching stats
        if mismatching == "matching":
            self.stats['matching_images'] += 1
        elif mismatching == "mismatching":
            self.stats['mismatching_images'] += 1
        else:
            raise ValueError(f"Filename parsing error for mismatching. Expected 'matching' or 'mismatching', got {mismatching}.")

        # Update positive/negative stats
        if pos_or_neg == "positive":
            self.stats['positive_images'] += 1
        elif pos_or_neg == "negative":
            self.stats['negative_images'] += 1
        else:
            raise ValueError(f"Filename parsing error for pos_or_neg. Expected 'positive' or 'negative', got {pos_or_neg}.")

        # Update emotion-specific stats
        self.stats[f'gt-{gt_emotion}_images'] += 1
        self.stats[f'occ-{occ_emotion}_images'] += 1

    def check_consistency(self):
        # Check if processed images match skipped + saved images
        if self.stats['processed_images'] != (self.stats['skipped_images'] + self.stats['saved_images']):
            print(f"[WARNING] Processed images ({self.stats['processed_images']}) != "
                  f"Skipped images ({self.stats['skipped_images']}) + Saved images ({self.stats['saved_images']})")
            print(str(self))
            
    def increase_processed(self):
        self.stats['processed_images'] += 1

    def increase_skipped(self):
        self.stats['skipped_images'] += 1

    def increase_saved(self):
        self.stats['saved_images'] += 1

    

# ============================ End of Highly Coupled Functions =============================

