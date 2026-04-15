import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH
import h5py
import numpy as np

if __name__ == "__main__":
    # Load the original h5 file
    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, 'r') as f:
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
        class_names = f['class_names'][:]
        paths = f['paths'][:]

    # Update paths: prepend image_{idx:03d}__ to the basename of each path
    new_paths = []
    for idx, path in enumerate(paths):
        path_str = path.decode('utf-8')
        basename = os.path.basename(path_str)
        new_basename = f"image_{idx:03d}__{basename}"
        new_path = os.path.join(os.path.dirname(path_str), new_basename)
        new_paths.append(new_path.encode('utf-8'))

    new_paths = np.array(new_paths, dtype=object)

    # Save to new h5 file
    with h5py.File(OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH, 'w') as f:
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.create_dataset('class_names', data=class_names)
        f.create_dataset('paths', data=new_paths)

    print("Saved updated h5 file to:", OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH)

    # Verify new h5 contents
    with h5py.File(OCCLUDED_TEST_SET_H5_NEWFILENAMES_PATH, 'r') as f:
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
        class_names = f['class_names'][:]
        paths = f['paths'][:]

    print(f"X_test.shape: {X_test.shape}")
    print(f"X_test dtype: {X_test.dtype}")
    print(f"class_names.shape: {class_names.shape}")
    print(f"class_names: {class_names}")
    print(f"paths.shape: {paths.shape}")
    print(f"paths (first and last five): {paths[:5]} ... {paths[-5:]}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"y_test: {y_test}")

    # ---- Check all path basenames exist in OCCLUDED_TEST_SET_RESIZED_PATH ----
    existing_files = set()
    for _, _, files in os.walk(OCCLUDED_TEST_SET_RESIZED_PATH):
        existing_files.update(files)

    missing = []
    for p in paths:
        basename = os.path.basename(p.decode('utf-8'))
        if basename not in existing_files:
            missing.append(basename)

    if missing:
        print(f"\n❌ {len(missing)}/{len(paths)} paths NOT found in {OCCLUDED_TEST_SET_RESIZED_PATH}:")
        for m in missing:
            print(f"  - {m}")
    else:
        print(f"\n✅ All {len(paths)} paths found in {OCCLUDED_TEST_SET_RESIZED_PATH}")
    # ---- End check ----

    exit(0)