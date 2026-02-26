import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import h5py;
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np

from modules.config import OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_IMAGES_PATH
from modules.misc import hash_image



def load_h5_dataset(h5_path):
    """Load the H5 dataset and return images, labels, paths, and class names."""
    with h5py.File(h5_path, 'r') as h5_file:
        X_test = h5_file['X_test'][:]
        class_names = h5_file['class_names'][:]
    return X_test, class_names


def hash_images_dict(images):
    """Hash all images in the extracted folder."""
    hashes = {idx: hash_image(image) for idx, image in images.items()}
    return hashes


def compare_hashes(h5_hashes, extracted_hashes):
    """Compare hashes from the H5 dataset and extracted folder."""
    mismatches = {}

    for idx, h5_hash in h5_hashes.items():
        extracted_hash = extracted_hashes.get(idx)
        if h5_hash != extracted_hash:
            mismatches[idx] = (h5_hash, extracted_hash)
    return mismatches


def show_mismatches(X_test, extracted_images, mismatches, rows=6, couples_per_row=2):
    """Visualize mismatched images in multiple plots if necessary."""
    cols = couples_per_row * 2  # Two images per couple (H5 and extracted)
    total_images_per_plot = rows * cols
    mismatch_indices = list(mismatches.keys())

    num_plots = (len(mismatch_indices) + (total_images_per_plot // 2) - 1) // (total_images_per_plot // 2)

    for plot_idx in range(num_plots):
        start_idx = plot_idx * (total_images_per_plot // 2)
        end_idx = start_idx + (total_images_per_plot // 2)
        current_indices = mismatch_indices[start_idx:end_idx]

        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten()

        for i, idx in enumerate(current_indices):
            h5_image = X_test[idx]
            extracted_image = extracted_images[idx]

            # Plot H5 image
            axes[2 * i].imshow(h5_image.astype(np.uint8))
            axes[2 * i].set_title(f"H5 Image (Idx: {idx})")
            axes[2 * i].axis('off')

            # Plot Extracted image
            axes[2 * i + 1].imshow(extracted_image)
            axes[2 * i + 1].set_title(f"Extracted Image (Idx: {idx})")
            axes[2 * i + 1].axis('off')

        # Hide unused axes
        for j in range(2 * len(current_indices), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


def load_extracted_folder_images(folder_path, extract_idx_function=None):
    """Load all images from the extracted folder."""
    i = 0
    images = {}
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):  # Ensure consistent order
            file_path = os.path.join(root, file)
            image = Image.open(file_path)

            if extract_idx_function is not None:
                idx = extract_idx_function(file_path)
            else:                
                idx = i  # Default to sequential index if no function provided

            images[idx] = np.array(image)
            i += 1
    return images


def check_presence_at_other_index(h5_hashes, extracted_hashes, mismatches):
    """Check if mismatched H5 hashes are present at a different index in the extracted hashes."""
    for idx, (h5_hash, _) in mismatches.items():
        found_at_index = None
        for ext_idx, ext_hash in extracted_hashes.items():
            if h5_hash == ext_hash:
                found_at_index = ext_idx
                break
        if found_at_index is not None:
            print(f"H5 hash at index {idx:3d} found in extracted images at index {found_at_index}.")
        else:
            print(f"H5 hash at index {idx:3d} not found in any extracted image.")

    for idx, (_, ext_hash) in mismatches.items():
        found_at_index = None
        for h5_idx, h5_hash in h5_hashes.items():
            if ext_hash == h5_hash:
                found_at_index = h5_idx
                break
        if found_at_index is not None:
            print(f"Extracted hash at index {idx:3d} found in H5 images at index {found_at_index}.")
        else:
            print(f"Extracted hash at index {idx:3d} not found in any H5 image.")


def do_check(X_test, h5_hashes, extracted_images_folder_path, extract_idx_function):
    # Load extracted folder images
    extracted_images = load_extracted_folder_images(extracted_images_folder_path, extract_idx_function)
    extracted_hashes = hash_images_dict(extracted_images)

    # Compare hashes
    mismatches = compare_hashes(h5_hashes, extracted_hashes)
    print(f"Found {len(mismatches)} mismatches.")

    if len(mismatches) > 0:
        print("Mismatches found at indices:")
        for idx in mismatches.keys():
            print(f"\tIndex {idx}: H5 hash {mismatches[idx][0]} != Extracted hash {mismatches[idx][1]}")

        check_presence_at_other_index(h5_hashes, extracted_hashes, mismatches)

        # show_mismatches(X_test, extracted_images, mismatches)

    return mismatches


def extract_idx_adele_test_set(filename):
    """Extract the index from the ADELE test set filename."""
    # Example filename: image_0_Happy_abc123.png
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    if len(parts) == 4 and parts[0] == 'image':
        return int(parts[1])  # Extract the index part
    raise ValueError(f"Filename does not match expected format: {filename}")


def extract_idx_occluded_test_set(filename):
    """Extract the index from the ADELE test set filename."""
    # Example filename: image_0_Happy_abc123.png
    base_name = os.path.basename(filename)
    parts__ = base_name.split('__')
    image_part = parts__[0]  # e.g. image_0_Happy_abc123.png
    if not parts__[-1].endswith('.png') or not image_part.startswith('image_'):
        raise ValueError(f"Filename does not match expected format wrt image_ and .png: {filename}")
    
    return int(image_part.split('_')[1])  # Extract the index part from image_0_Happy_abc123.png -> 0

# ========================================== SETTINGS ============================================  

argparser = argparse.ArgumentParser(description='Check occluded images dataset against extracted images.')
argparser.add_argument('--dataset', type=str, choices=['occluded', 'adele'], default='occluded', help='Dataset to check (default: occluded)')
args = argparser.parse_args()

if args.dataset == 'occluded':
    H5_FILE_PATH = OCCLUDED_TEST_SET_H5_PATH
    EXTRACTED_IMAGES_FOLDER_PATH = OCCLUDED_TEST_SET_RESIZED_PATH
    EXTRACT_IDX_FUNCTION = extract_idx_occluded_test_set
elif args.dataset == 'adele':
    H5_FILE_PATH = ADELE_TEST_SET_H5_PATH
    EXTRACTED_IMAGES_FOLDER_PATH = ADELE_TEST_SET_IMAGES_PATH
    EXTRACT_IDX_FUNCTION = extract_idx_adele_test_set

print(f"=========== SETTINGS ============")
print(f"ARGS:")
print(f"\tDataset: {args.dataset}")
print(f"Paths:")
print(f"\tH5 file: {H5_FILE_PATH}")
print(f"\tExtracted images folder: {EXTRACTED_IMAGES_FOLDER_PATH}")

# =================================================================================================


if __name__ == '__main__':
    # Load H5 dataset
    X_test, class_names = load_h5_dataset(H5_FILE_PATH)
    X_test = {idx: image for idx, image in enumerate(X_test)}  # Convert to dict for easier access by index
    h5_hashes = hash_images_dict(X_test)

    mismatches_noextract = do_check(X_test, h5_hashes, EXTRACTED_IMAGES_FOLDER_PATH, None)

    if EXTRACT_IDX_FUNCTION is not None:
        mismatches_withextract = do_check(X_test, h5_hashes, EXTRACTED_IMAGES_FOLDER_PATH, EXTRACT_IDX_FUNCTION)

        if len(mismatches_noextract) == len(mismatches_withextract):
            print(f"[INFO] [IMPORTANT] Extracting index from filename did not change the number of mismatches. ({len(mismatches_noextract)} mismatches)")
        else:
            print(f"[INFO] [IMPORTANT] Extracting index from filename changed the number of mismatches from {len(mismatches_noextract)} to {len(mismatches_withextract)}.")

        
        for (key_noextract, value_noextract), (key_extract, value_extract) in zip(mismatches_noextract.items(), mismatches_withextract.items()):
            if key_noextract != key_extract or value_noextract != value_extract:
                print(f"Mismatch changed for index {key_noextract}:")
                print(f"\tWithout extract: {value_noextract}")
                print(f"\tWith extract: {value_extract}")

        if len(mismatches_noextract) == len(mismatches_withextract):
            print(f"[INFO] [IMPORTANT] I repeat: Extracting index from filename did not change the number of mismatches. ({len(mismatches_noextract)} mismatches)")
        else:
            print(f"[INFO] [IMPORTANT] I repeat: Extracting index from filename changed the number of mismatches from {len(mismatches_noextract)} to {len(mismatches_withextract)}.")
