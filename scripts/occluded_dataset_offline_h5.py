import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from modules.config import EMOTIONS, OCCLUDED_TRAIN_SET_IMAGES_PATH, OCCLUDED_VAL_SET_IMAGES_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, IMAGES_SHAPE
from modules.misc import extract_info_from_occludedtrainvalset_filename

# Paths
TRAINSET_PATH = OCCLUDED_TRAIN_SET_IMAGES_PATH
VALSET_PATH = OCCLUDED_VAL_SET_IMAGES_PATH
OUTPUT_H5_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH

# Image dimensions (assumed to be consistent across the dataset)
IMG_HEIGHT = IMAGES_SHAPE[0]
IMG_WIDTH = IMAGES_SHAPE[1]
IMG_CHANNELS = IMAGES_SHAPE[2]

parser = argparse.ArgumentParser(description="Create HDF5 dataset for occluded emotion recognition.")
#                                                     2**15
parser.add_argument("--batch_size", type=int, default=32768, help="Batch size for processing images.")

args = parser.parse_args()


def process_batch(base_dir, batch_size, h5_file, dataset_prefix):
    """
    Process images in batches and write them to the HDF5 file.
    """
    images, labels, original_hashes, occ_labels, mismatches, pos_or_negs = [], [], [], [], [], []
    total_images = sum(len(files) for _, _, files in os.walk(base_dir))
    current_index = 0

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for image_name in tqdm(os.listdir(folder_path), desc=f"Processing {dataset_prefix} data. Browsing folder {folder}"):
            original_hash, gt_emotion, occ_emotion, mismatching, pos_or_neg = extract_info_from_occludedtrainvalset_filename(image_name)
            gt_emotion_label = EMOTIONS.index(gt_emotion.upper())
            occ_emotion_label = EMOTIONS.index(occ_emotion.upper())

            image_path = os.path.join(folder_path, image_name)
            try:
                # Load and resize the image
                image = Image.open(image_path).convert("RGB")
                image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                images.append(np.array(image, dtype=np.uint8))

                labels.append(gt_emotion_label)
                occ_labels.append(occ_emotion_label)
                mismatches.append(0 if mismatching == "matching" else 1)
                pos_or_negs.append(0 if pos_or_neg == "negative" else 1)
                original_hashes.append(original_hash)
                current_index += 1

                # If the batch is full, write to HDF5 and reset the batch
                if len(images) == batch_size or current_index == total_images:
                    write_batch_to_h5(h5_file, dataset_prefix, images, labels, original_hashes, occ_labels, mismatches, pos_or_negs, current_index - len(images))
                    images, labels, original_hashes, occ_labels, mismatches, pos_or_negs = [], [], [], [], [], []

            except Exception as e:
                print(f"Error loading image {image_path}: {e}")


def write_batch_to_h5(h5_file, dataset_prefix, images, labels, original_hashes, occ_labels, mismatches, pos_or_negs, start_index):
    """
    Write a batch of data to the HDF5 file.
    """
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    occ_labels = np.array(occ_labels, dtype=np.uint8)
    mismatches = np.array(mismatches, dtype=np.uint8)
    pos_or_negs = np.array(pos_or_negs, dtype=np.uint8)

    h5_file[f"X_{dataset_prefix}"]  [start_index : start_index + len(images)] = images
    h5_file[f"y_{dataset_prefix}"][start_index : start_index + len(labels)] = labels
    h5_file[f"original_hash_{dataset_prefix}"][start_index : start_index + len(original_hashes)] = np.array(original_hashes, dtype='S32')
    h5_file[f"occ_{dataset_prefix}"][start_index : start_index + len(occ_labels)] = occ_labels
    h5_file[f"mismatch_{dataset_prefix}"][start_index : start_index + len(mismatches)] = mismatches
    h5_file[f"pos_or_neg_{dataset_prefix}"][start_index : start_index + len(pos_or_negs)] = pos_or_negs

def create_h5_datasets(output_path, train_dir, val_dir, batch_size):
    """
    Create HDF5 datasets for training and validation data.
    """
    # Count total images in train and validation sets
    total_train_images = sum(len(files) for _, _, files in os.walk(train_dir))
    total_val_images = sum(len(files) for _, _, files in os.walk(val_dir))

    with h5py.File(output_path, "w") as h5_file:
        # Create datasets with the total size pre-allocated
        h5_file.create_dataset("X_train", shape=(total_train_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype="uint8")
        h5_file.create_dataset("y_train", shape=(total_train_images,), dtype="uint8")
        h5_file.create_dataset("original_hash_train", shape=(total_train_images,), dtype="S32")
        h5_file.create_dataset("occ_train", shape=(total_train_images,), dtype="uint8")
        h5_file.create_dataset("mismatch_train", shape=(total_train_images,), dtype="uint8")
        h5_file.create_dataset("pos_or_neg_train", shape=(total_train_images,), dtype="uint8")

        h5_file.create_dataset("X_val", shape=(total_val_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype="uint8")
        h5_file.create_dataset("y_val", shape=(total_val_images,), dtype="uint8")
        h5_file.create_dataset("original_hash_val", shape=(total_val_images,), dtype="S32")
        h5_file.create_dataset("occ_val", shape=(total_val_images,), dtype="uint8")
        h5_file.create_dataset("mismatch_val", shape=(total_val_images,), dtype="uint8")
        h5_file.create_dataset("pos_or_neg_val", shape=(total_val_images,), dtype="uint8")

        # add class names
        h5_file.create_dataset("class_names", data=np.array(EMOTIONS, dtype='S'))

        # Process training data in batches
        print("Processing training data...")
        process_batch(train_dir, batch_size, h5_file, "train")

        # Process validation data in batches
        print("Processing validation data...")
        process_batch(val_dir, batch_size, h5_file, "val")

    print(f"HDF5 dataset saved to {output_path}")


if __name__ == "__main__":
    # Create HDF5 datasets
    create_h5_datasets(OUTPUT_H5_PATH, TRAINSET_PATH, VALSET_PATH, args.batch_size)