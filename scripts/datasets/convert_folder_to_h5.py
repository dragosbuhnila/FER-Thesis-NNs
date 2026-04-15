import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import h5py
import numpy as np
from PIL import Image
from typing import Optional, Sequence, Tuple

from modules.config import BOSPHORUS_TEST_HQ_H5_PATH, BOSPHORUS_TEST_HQ_IMAGES_PATH, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, \
                            ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH



# ================================== MACROS ==================================

JUST_CHECK_RESULT = True
IMAGES_FOLDER_PATH = BOSPHORUS_TEST_HQ_IMAGES_PATH # OCCLUDED_TEST_SET_PATH, BOSPHORUS_TEST_FINALE, BOSPHORUS_TEST_HQ_IMAGES_PATH
RESIZE_TO_SMALL = False
H5_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH  # OCCLUDED_TEST_SET_H5_PATH, EXAMPLE_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH

# =============================== END OF MACROS ===============================

def create_h5_from_folder(images_folder_path: str,
                          h5_path: str,
                          resize_to: Optional[Tuple[int,int]] = None,
                          class_names: Optional[Sequence[str]] = None,
                          save_paths: bool = True,
                          compress: bool = True) -> dict:
    """Create an HDF5 dataset from a folder of class subfolders.

    - images_folder_path: root folder containing one folder per class.
    - h5_path: output .h5 file path.
    - resize_to: (w,h) to resize images, or None to keep original sizes.
    - class_names: optional explicit order of class folders; if None uses sorted directories.
    - save_paths: save original image paths into HDF5.
    - compress: use gzip compression for image dataset.
    Returns a summary dict with counts and class_names.
    """
    if class_names is None:
        class_names = [d for d in sorted(os.listdir(images_folder_path))
                       if os.path.isdir(os.path.join(images_folder_path, d))]
    X, y, paths = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(images_folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in sorted(os.listdir(class_folder)):
            img_path = os.path.join(class_folder, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                if resize_to:
                    img = img.resize(resize_to)
                arr = np.array(img)
            except Exception:
                continue
            X.append(arr)
            y.append(class_idx)
            paths.append(img_path)
    if len(X) == 0:
        raise ValueError("No images found in folder: " + images_folder_path)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    with h5py.File(h5_path, "w") as f:
        kwargs = {"data": X}
        if compress:
            f.create_dataset("X", data=X, compression="gzip")
        else:
            f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))
        if save_paths:
            f.create_dataset("paths", data=np.array(paths).astype('S'))
    return {"n_images": X.shape[0], "class_names": class_names}


if __name__ == "__main__":
    resize = (128,128) if RESIZE_TO_SMALL else None
    summary = create_h5_from_folder(IMAGES_FOLDER_PATH, H5_PATH, resize_to=resize)
    print(f"Saved {summary['n_images']} images to {H5_PATH}")