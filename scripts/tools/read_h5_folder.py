import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import h5py

from modules.config import BOSPHORUS_TEST_HQ_H5_PATH, BOSPHORUS_TEST_HQ_IMAGES_PATH, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, \
                            ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH



# ================================== MACROS ==================================

H5_PATHS = [  # OCCLUDED_TEST_SET_H5_PATH, EXAMPLE_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
    ORIGINAL_TRAIN_VAL_SET_H5_PATH,
    ADELE_TEST_SET_H5_PATH,
]

# =============================== END OF MACROS ===============================



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
        print("======================")