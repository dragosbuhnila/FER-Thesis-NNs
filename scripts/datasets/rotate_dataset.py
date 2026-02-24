import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) 

import h5py
import numpy as np
from PIL import Image

from modules.config import ADELE_TEST_SET_H5_PATH, ADELE_180ROTATED_TEST_SET_H5_PATH, ADELE_180ROTATED_TEST_SET_IMAGES_PATH, EMOTIONS
from modules.misc import hash_image;

# Paths
original_h5_path = ADELE_TEST_SET_H5_PATH 
rotated_h5_path = ADELE_180ROTATED_TEST_SET_H5_PATH 
rotated_images_path = ADELE_180ROTATED_TEST_SET_IMAGES_PATH

# Ensure the output directory exists
os.makedirs(os.path.dirname(rotated_h5_path), exist_ok=True)
os.makedirs(rotated_images_path, exist_ok=True)



print(f"========================================== SETTINGS ==========================================")
print(f"Original H5 path: {original_h5_path}")
print(f"Rotated H5 path: {rotated_h5_path}")
print(f"Rotated images path: {rotated_images_path}")
print(f"============================================================================================")



if __name__ == "__main__":
    # Open the original H5 file
    with h5py.File(original_h5_path, "r") as original_h5:
        # Read the test set data
        X_test = np.array(original_h5["X_test"])
        y_test = np.array(original_h5["y_test"])
        class_names = np.array(original_h5["class_names"])

        # Rotate all images by 180 degrees
        X_hashes = [hash_image(img) for img in X_test]
        X_hashes = np.array([str(h).encode('utf-8') for h in X_hashes], dtype='S')  # 'S' ensures fixed-length strings
        X_test_rotated = np.rot90(X_test, k=2, axes=(1, 2))  # Rotate 180 degrees (2 * 90 degrees)

        # Save the rotated dataset to a new H5 file
        with h5py.File(rotated_h5_path, "w") as rotated_h5:
            rotated_h5.create_dataset("X_test", data=X_test_rotated, dtype=X_test.dtype)
            rotated_h5.create_dataset("y_test", data=y_test, dtype=y_test.dtype)
            rotated_h5.create_dataset("class_names", data=class_names, dtype=class_names.dtype)
            rotated_h5.create_dataset("X_hashes", data=X_hashes)

        # Save the rotated images to the specified directory
        for image, hash, gt in zip(X_test_rotated, X_hashes, y_test):
            label = EMOTIONS[gt]
            # TODO: convert b'' hash to str
            image_path = os.path.join(rotated_images_path, "label", f"{hash}_180rotated.png")
            Image.fromarray(image).save(image_path)

    print(f"[INFO] Rotated test set created.")
    print(f"[INFO] Rotated H5 file saved at: {rotated_h5_path}")
    print(f"[INFO] Rotated images saved in: {rotated_images_path}")