import os
import h5py
import numpy as np
from PIL import Image

from modules.config import OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_PATH, OCCLUDED_TEST_SET_RESIZED_PATH

EXAMPLE_H5_PATH = os.path.join(".", "saliency_maps", "zzz_other_and_zips", "h5_files", "test_data_adele.h5")
# test set folder contains subfolders for each class (ANGRY, DISGUST, ...) with inside images called like "bosphorus_bs001_ANGRY_30__masked-negative-DISGUST_mismatch.png"

JUST_CHECK_RESULT = True

if __name__ == "__main__":
    # 2) Generate new h5 with the contents of the test set
    class_names = sorted(os.listdir(OCCLUDED_TEST_SET_PATH))
    paths = []
    X_test = []
    y_test = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(OCCLUDED_TEST_SET_PATH, class_name)
        image_files = sorted(os.listdir(class_folder))
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            # Load PNG image as RGB NumPy array and resize to (128, 128, 3)
            image = np.array(Image.open(image_path).convert('RGB').resize((128, 128)))
            X_test.append(image)
            y_test.append(class_idx)  # Store class index instead of name
            paths.append(image_path)

            # Also save the images to OCCLUDED_TEST_SET_RESIZED_PATH
            save_folder = os.path.join(OCCLUDED_TEST_SET_RESIZED_PATH, class_name)
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(save_folder, image_file))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 3) Save new h5
    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, "w") as f:
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)  # Now integers
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))  # Save as bytes
        f.create_dataset("paths", data=np.array(paths).astype('S'))  # Save as bytes
    print(f"Saved {X_test.shape[0]} images to {OCCLUDED_TEST_SET_H5_PATH}")

    # 4) Verify
    print("======================")
    print("Verifying new h5 contents:")
    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, "r") as f:
        for key in f.keys():
            print(f"{key}: {f[key].shape}")
            if key != "X_test":
                if key == "paths":
                    print(f"{key} (first and last five): {f[key][:5]} ... {f[key][-5:]}")
                else:
                    print(f"{key}: {f[key][...]}")
            # result:
            # X_test: (350, 128, 128, 3)
            # y_test: (350,)
            # class_names: (7,)
    print("======================")