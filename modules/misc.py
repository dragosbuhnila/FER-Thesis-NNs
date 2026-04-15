import os
import sys
import zipfile
from PIL import Image
import h5py
import numpy as np
import hashlib
import time
from typing import Optional, Sequence, Tuple
from modules.config import (
    ADELE_180ROTATED_TEST_SET_H5_PATH,
    ADELE_180ROTATED_TEST_SET_IMAGES_PATH,
    ADELE_TEST_SET_H5_PATH,
    ADELE_TEST_SET_IMAGES_PATH,
    OCCLUDED_TEST_SET_H5_PATH,
    OCCLUDED_TEST_SET_RESIZED_PATH,

    OCCLUDED_TEST_SET_H5_MATCHING_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_RESIZED_MATCHING_PATH,
    OCCLUDED_TEST_SET_H5_MISMATCHING_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_RESIZED_MISMATCHING_PATH,

    OCCLUDED_TEST_SET_H5_POSITIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_SAD_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_SAD_PATH,
    OCCLUDED_TEST_SET_H5_POSITIVE_SURPRISE_PATH,
    OCCLUDED_TEST_SET_RESIZED_POSITIVE_SURPRISE_PATH,

    OCCLUDED_TEST_SET_H5_NEGATIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_ANGRY_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_DISGUST_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_FEAR_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_HAPPY_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_SAD_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SAD_PATH,
    OCCLUDED_TEST_SET_H5_NEGATIVE_SURPRISE_PATH,
    OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SURPRISE_PATH,

    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MATCHING_RESIZED_IMAGES_PATH,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MISMATCHING_RESIZED_IMAGES_PATH,
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
    
    EMOTIONS,
    IMAGES_SHAPE,
    OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH,
)



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


def zip_folder(folder_path, output_path):
    """
    Zippa un’intera cartella 'folder_path' in un file 'output_path'.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=relative_path)


def extract_image_index_original_or_180rotated(filename):
    # check that the npys generated by the XAI methods for each image of the test set have the same numbering as the filanme of the original image processed.

    # like image_0_ANGRY_9c2bb200de89f096c0b8c19eed47fc22.png
    if not filename.startswith("image"):
        raise ValueError(f"Unexpected filename format: {filename}")
    
    no_ext_part = filename.split('.')[0]  # remove extension
    index_part = no_ext_part.split('_')[1]
    index = int(index_part)
    return index


def make_xai_result_image_name(path):
    # path looks like: b'.\\data\\datasets\\occluded_test_set\\bosphorus_test_HQ\\SURPRISE\\bosphorus_bs085_SURPRISE_472__masked-positive-ANGRY_mismatch.png'
    path = path.decode('utf-8') if isinstance(path, bytes) else path
    filename = os.path.basename(path)
    image_index = extract_image_index_original_or_180rotated(filename)
    return f"image_{image_index}"

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
# ====================================================================================================================================








# ====================================================================================================================================
# ============================ Print and Visualization Functions =====================================================================
# ========================= print_npy, create_placeholder_image ======================================================================
# ====================================================================================================================================

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


def create_placeholder_image(size=(224, 224), color=(255, 255, 255)):
    """
    Create a placeholder image of the given size and color.

    Parameters
    ----------
    size : tuple of int, optional
        The size of the placeholder image (width, height). Default is (224, 224).
    color : tuple of int, optional
        The RGB color of the placeholder image. Default is (255, 255, 255).

    Returns
    -------
    np.ndarray
        The placeholder image as a NumPy array.

    Examples
    --------
    >>> placeholder = create_placeholder_image()
    >>> placeholder.shape
    (224, 224, 3)
    """
    placeholder_img = Image.new('RGB', size, color=color)
    return np.array(placeholder_img)

# ====================================================================================================================================
# ============================ End of Print and Visualization Functions ==============================================================
# ====================================================================================================================================









# ====================================================================================================================================
# ============================ Highly Coupled Functions ==============================================================================
# ====================================================================================================================================

# ___________________________________________________________________________________________________________________
# do_evaluation_accuracy_simple.py
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

# _______________________________________________________________________________________________
# occlude_dataset_offline.py occlude_images.py calculate_expected_size_of_occluded_dataset.py
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
    

# ____________________________________________________________________________________________________________________
#   XAI
# ____________________________________________________________________________________________________________________
TEST_SET_CHOICES = [
    'occluded', 
    'original', 
    'original-180', 
    'occluded-matching', 
    'occluded-mismatching',
    'occlusion-positive-angry',
    'occlusion-positive-disgust',
    'occlusion-positive-fear',
    'occlusion-positive-happy',
    'occlusion-positive-sad',
    'occlusion-positive-surprise',
    'occlusion-negative-angry',
    'occlusion-negative-disgust',
    'occlusion-negative-fear',
    'occlusion-negative-happy',
    'occlusion-negative-sad',
    'occlusion-negative-surprise',
]

TEST_SET_PATHS = {
    'occluded': {
        'h5_path': OCCLUDED_TEST_SET_H5_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH,
    },
    'original': {
        'h5_path': ADELE_TEST_SET_H5_PATH,
        'images_path': ADELE_TEST_SET_IMAGES_PATH,
        'unoccluded_back_images_path': None,
    },
    'original-180': {
        'h5_path': ADELE_180ROTATED_TEST_SET_H5_PATH,
        'images_path': ADELE_180ROTATED_TEST_SET_IMAGES_PATH,
        'unoccluded_back_images_path': None,
    },
    'occluded-matching': {
        'h5_path': OCCLUDED_TEST_SET_H5_MATCHING_RESIZED_IMAGES_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_MATCHING_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MATCHING_RESIZED_IMAGES_PATH,
    },
    'occluded-mismatching': {
        'h5_path': OCCLUDED_TEST_SET_H5_MISMATCHING_RESIZED_IMAGES_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_MISMATCHING_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_MISMATCHING_RESIZED_IMAGES_PATH,
    },
    'occlusion-positive-angry': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_ANGRY_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_ANGRY_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_ANGRY_PATH,
    },
    'occlusion-positive-disgust': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_DISGUST_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_DISGUST_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_DISGUST_PATH,
    },
    'occlusion-positive-fear': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_FEAR_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_FEAR_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_FEAR_PATH,
    },
    'occlusion-positive-happy': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_HAPPY_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_HAPPY_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_HAPPY_PATH,
    },
    'occlusion-positive-sad': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_SAD_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_SAD_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_SAD_PATH,
    },
    'occlusion-positive-surprise': {
        'h5_path': OCCLUDED_TEST_SET_H5_POSITIVE_SURPRISE_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_POSITIVE_SURPRISE_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_POSITIVE_SURPRISE_PATH,
    },
    'occlusion-negative-angry': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_ANGRY_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_ANGRY_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_ANGRY_PATH,
    },
    'occlusion-negative-disgust': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_DISGUST_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_DISGUST_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_DISGUST_PATH,
    },
    'occlusion-negative-fear': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_FEAR_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_FEAR_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_FEAR_PATH,
    },
    'occlusion-negative-happy': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_HAPPY_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_HAPPY_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_HAPPY_PATH,
    },
    'occlusion-negative-sad': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_SAD_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SAD_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_SAD_PATH,
    },
    'occlusion-negative-surprise': {
        'h5_path': OCCLUDED_TEST_SET_H5_NEGATIVE_SURPRISE_PATH,
        'images_path': OCCLUDED_TEST_SET_RESIZED_NEGATIVE_SURPRISE_PATH,
        'unoccluded_back_images_path': OCCLUDED_TEST_SET_UNOCCLUDED_BACK_NEGATIVE_SURPRISE_PATH,
    },
}

def get_test_set_name_from_run_name(run_name):
    """
    Extract the test set name from the run name. Assumes the run name contains the test set name as a substring.
    """
    # Sort TEST_SET_CHOICES by length in descending order to prioritize longer names
    sorted_test_set_choices = sorted(TEST_SET_CHOICES, key=len, reverse=True)
    
    for test_set_name in sorted_test_set_choices:
        if test_set_name in run_name:
            return test_set_name
    raise ValueError(f"Could not extract test set name from run name: {run_name}. Expected one of: {TEST_SET_CHOICES}")

# ====================================================================================================================================
# ============================ End of Highly Coupled Functions =======================================================================
# ====================================================================================================================================
