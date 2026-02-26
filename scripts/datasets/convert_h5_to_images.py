import sys; import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import h5py
from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, ADELE_180ROTATED_TEST_SET_IMAGES_PATH, ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_IMAGES_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_IMAGES_BASE_PATH
from PIL import Image

from modules.misc import hash_image;



def _to_str(s):
    # convert bytes to str if needed
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return str(s)


def extract_images_from_h5(h5_path, output_base, dont_hash, group_keys=None, overwrite=False, verbose=True):
    """Extract image arrays from H5 into folders named after class names.

    - h5_path: path to h5 file
    - output_base: directory where extracted images will be stored (will be created)
    - group_keys: list of dataset suffixes to extract (e.g. ['test','train','val']) or None to autodetect pairs
    - overwrite: whether to overwrite existing files
    """
    os.makedirs(output_base, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        # discover class_names
        if 'class_names' not in f:
            raise RuntimeError('H5 file missing "class_names" dataset')
        class_names = [ _to_str(x) for x in list(f['class_names'][...]) ]

        # autodetect X/y pairs if not provided
        if group_keys is None:
            hashes_key = None
            groups = []
            # find X_* and corresponding y_*
            for key in f.keys():
                if key.startswith('X_'):
                    suffix = key[2:]
                    y_key = f'y_{suffix}'
                    if y_key in f:
                        groups.append(suffix)
                if "hash" in key:
                    hashes_key = key
            if not groups:
                raise RuntimeError('No X_/y_ pairs found in H5 file')
            if len(groups) > 1 and verbose:
                print(f'[WARNING] So far autodetected groups: {groups}. But group_keys was not None, why? group_keys: {group_keys}')
        else:
            groups = group_keys

        summary = {}

        if hashes_key:
            hashes = f[hashes_key]

        if hashes_key and dont_hash:
            print(f"[INFO] hash key found as {hashes_key}.")
        elif not hashes_key and dont_hash:
            print(f"[WARNING] dont_hash specified, but found no hashing key in the dataset. Hashes will not be put in the image file names")
        elif hashes_key and not dont_hash:
            print(f"[WARNING] dont_hash is False, but found a hash key in the dataset as {hashes_key}. Are you sure you want to rehash the image? It could be an augmented image!")

        nof_groups = len(groups)
        for suffix in groups:
            X_key = f'X_{suffix}'
            y_key = f'y_{suffix}'
            if X_key not in f or y_key not in f:
                if verbose:
                    print(f'Skipping missing pair: {X_key} / {y_key}')
                continue

            X = f[X_key]
            y = f[y_key]

            n = X.shape[0]
            if n > 999:
                raise RuntimeError(f"Too many images ({n}) to rename with 3 digits. Please adjust the code to use more digits if needed.")
            out_dir = os.path.join(output_base, suffix) if nof_groups > 1 else output_base
            os.makedirs(out_dir, exist_ok=True)

            if verbose:
                print(f'Extracting {n} images for group "{suffix}" into {out_dir}')

            # We'll create per-class subfolders and name files with a global index matching typical convention
            counts = {name: 0 for name in class_names}

            for i in range(n):
                img = X[i]
                label = int(y[i])
                class_name = class_names[label]
                class_folder = os.path.join(out_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)

                # filename: image_{global_idx}_{gt}_{hash}.png so ordering mirrors previous folder structure except for the addition of the hashing
                if dont_hash:
                    if hashes:
                        img_hash = hashes[i]
                    else:
                        img_hash = ""
                else:
                    img_hash = hash_image(img)
                gt = class_name
                fname = f'image_{i:03d}_{gt}_{img_hash}.png'
                out_path = os.path.join(class_folder, fname)

                if not overwrite and os.path.exists(out_path):
                    print(f"[WARNING] File already exists and overwrite is False, skipping: {out_path}")
                    pass
                else:
                    # img is expected uint8 HWC
                    try:
                        Image.fromarray(img).save(out_path)
                    except Exception:
                        # try converting to uint8
                        arr = img.astype('uint8')
                        Image.fromarray(arr).save(out_path)

                counts[class_name] += 1

            summary[suffix] = counts
            if verbose:
                print('Extracted counts:', counts)

    return summary


parser = argparse.ArgumentParser(description='Extract images from H5 file into class-named folders.')
parser.add_argument('--dataset', type=str, choices=["ORIGINAL_TRAINVAL", "ADELE_TEST_SET", "180ADELE"], default="ADELE_TEST_SET", help='Which dataset to extract (determines which H5 path and output base to use)')
args = parser.parse_args()

if args.dataset == "ORIGINAL_TRAINVAL":
    H5_PATH =           ORIGINAL_TRAIN_VAL_SET_H5_PATH
    OUTPUT_BASE_PATH =  ORIGINAL_TRAIN_VAL_SET_IMAGES_BASE_PATH
    DONT_HASH_EXTRACT_IT_FROM_DATASET = False
elif args.dataset == "ADELE_TEST_SET":
    H5_PATH =           ADELE_TEST_SET_H5_PATH
    OUTPUT_BASE_PATH =  ADELE_TEST_SET_IMAGES_PATH
    DONT_HASH_EXTRACT_IT_FROM_DATASET = False
elif args.dataset == "180ADELE":
    H5_PATH =           ADELE_180ROTATED_TEST_SET_H5_PATH
    OUTPUT_BASE_PATH =  ADELE_180ROTATED_TEST_SET_IMAGES_PATH
    DONT_HASH_EXTRACT_IT_FROM_DATASET = True
else:
    raise ValueError(f"Unexpected dataset: {args.dataset}")


print("======================= SETTINGS =======================")
print("ARGS:")
print(f"\tDataset: {args.dataset}")
print(f"MACROS:")
print(f"\tH5_PATH: {H5_PATH}")
print(f"\tOUTPUT_BASE_PATH: {OUTPUT_BASE_PATH}")
print(f"\DONT_HASH_EXTRACT_IT_FROM_DATASET: {DONT_HASH_EXTRACT_IT_FROM_DATASET}")
print("CURRENT FORMAT FOR OUTPUT IMAGES: .../{gt}/image_{global_idx}_{gt}_{img_hash}.png")
print("========================================================")


# Example usage:
# >>> adele test set:
# `& C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/datasets/convert_h5_to_images.py" --dataset ADELE_TEST_SET`
# >>> original trainval set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/datasets/convert_h5_to_images.py" --dataset ORIGINAL_TRAINVAL
# >>> 180 rotated adele test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/datasets/convert_h5_to_images.py" --dataset 180ADELE
if __name__ == "__main__":
    # 1) Check if H5 file exists
    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(f"H5 file not found: {H5_PATH}")
    
    # 2) Check if extracted images don't already exist (if not overwriting)
    if os.path.exists(OUTPUT_BASE_PATH):
        if os.listdir(OUTPUT_BASE_PATH):
            print(f"[WARNING] Output base path already exists and is not empty: {OUTPUT_BASE_PATH}")
            print("\tContents:", os.listdir(OUTPUT_BASE_PATH))
            print("\tProceeding with extraction, but be aware of potential overwriting or mixing of old/new data.")


    # print(f"H5 to convert path: {H5_PATH}")
    # print("======================")
    # print("Verifying h5 contents:")
    # with h5py.File(H5_PATH, "r") as f:
    #     for key in f.keys():
    #         try:
    #             shape = f[key].shape
    #         except Exception:
    #             shape = '(unknown)'
    #         print(f"{key}.shape: {shape}")
    #         if "X" not in key:
    #             if key == "paths":
    #                 print(f"{key} (first and last five): {f[key][:5]} ... {f[key][-5:]}")
    #             else:
    #                 # careful printing large arrays
    #                 val = f[key][...]
    #                 print(f"{key}: {val}")
    #         else:
    #             print(f"{key} dtype: {f[key].dtype}")
    # print("======================")

    # Example: extract X_test / y_test into data/datasets/<h5basename>_extracted
    print('Default extraction output base:', OUTPUT_BASE_PATH)
    summary = extract_images_from_h5(H5_PATH, OUTPUT_BASE_PATH, dont_hash=DONT_HASH_EXTRACT_IT_FROM_DATASET)
    print('Extraction summary:', summary)