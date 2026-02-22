import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import h5py

from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, BOSPHORUS_TEST_HQ_H5_PATH, BOSPHORUS_TEST_HQ_IMAGES_PATH, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, \
                            ADELE_TEST_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, EMOTIONS

# ================================== MACROS ==================================

H5_PATHS = [ 
    # OCCLUDED_TRAIN_VAL_SET_H5_PATH,
    # OCCLUDED_TEST_SET_H5_PATH,
    # ORIGINAL_TRAIN_VAL_SET_H5_PATH,
    # ADELE_TEST_SET_H5_PATH,
    ADELE_180ROTATED_TEST_SET_H5_PATH
]

SHOW_IMAGES = False
COLS, ROWS = 8, 4  # 8 * 4 = 32 images per grid, adjust as needed

# =============================== END OF MACROS ===============================

def show_images_grid(images, labels, title="Images"):
    """
    Display a grid of images with their labels using matplotlib, iterating through the entire dataset.
    """
    total_images = len(images)
    batch_size = ROWS * COLS  # Number of images per grid

    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        fig, axes = plt.subplots(ROWS, COLS, figsize=(12, 6))
        fig.suptitle(f"{title} (Images {start_idx + 1}-{end_idx})", fontsize=16)
        for i, ax in enumerate(axes.flat):
            if i < len(batch_images):
                ax.imshow(batch_images[i])
                ax.axis('off')
                ax.set_title(EMOTIONS[batch_labels[i]], fontsize=8)  # Set the label as the title
            else:
                ax.axis('off')  # Hide unused subplots
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust to make space for the title
        plt.show()

if __name__ == "__main__":
    for H5_PATH in H5_PATHS:
        print("======================")
        print("Verifying new h5 contents:")
        with h5py.File(H5_PATH, "r") as f:
            for key in f.keys():
                try:
                    shape = f[key].shape
                except Exception:
                    shape = '(unknown)'
                print(f"{key}.shape: {shape}")
                if "X" not in key:
                    if key == "paths":
                        print(f"{key} (first and last five): {f[key][:5]} ... {f[key][-5:]}")
                    else:
                        # careful printing large arrays
                        val = f[key][...]
                        print(f"{key}: {val}")
                else:
                    print(f"{key} dtype: {f[key].dtype}")
                    if SHOW_IMAGES:
                        # Display images in a 3x6 grid
                        images = f[key][...]
                        label_key = key.replace("X", "y")  # Assume corresponding labels are stored as y_<split>
                        if label_key in f.keys():
                            labels = f[label_key][...]
                            if images.ndim == 4 and images.shape[-1] in [1, 3]:  # Ensure it's image data
                                show_images_grid(images, labels, title=f"Dataset: {key}")
                            else:
                                print(f"Skipping {key} as it does not appear to be image data.")
                        else:
                            print(f"Labels for {key} not found. Skipping visualization.")
        print("======================")