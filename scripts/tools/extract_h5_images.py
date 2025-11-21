import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
from modules.config import ORIGINAL_TRAIN_VAL_SET_H5_PATH
from PIL import Image

H5_TO_CONVERT_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH


def _to_str(s):
    # convert bytes to str if needed
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return str(s)


def extract_images_from_h5(h5_path, output_base, group_keys=None, overwrite=False, verbose=True):
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
            groups = []
            # find X_* and corresponding y_*
            for key in f.keys():
                if key.startswith('X_'):
                    suffix = key[2:]
                    y_key = f'y_{suffix}'
                    if y_key in f:
                        groups.append(suffix)
            if not groups:
                raise RuntimeError('No X_/y_ pairs found in H5 file')
        else:
            groups = group_keys

        summary = {}

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
            out_dir = os.path.join(output_base, suffix)
            os.makedirs(out_dir, exist_ok=True)

            if verbose:
                print(f'Extracting {n} images for group "{suffix}" into {out_dir}')

            # We'll create per-class subfolders and name files with a global index matching typical convention
            global_idx = 0
            counts = {name: 0 for name in class_names}

            for i in range(n):
                img = X[i]
                label = int(y[i])
                class_name = class_names[label]
                class_folder = os.path.join(out_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)

                # filename: image_{global_idx}.png so ordering mirrors previous folder structure
                fname = f'image_{global_idx}.png'
                out_path = os.path.join(class_folder, fname)

                if not overwrite and os.path.exists(out_path):
                    # skip writing
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
                global_idx += 1

            summary[suffix] = counts
            if verbose:
                print('Extracted counts:', counts)

    return summary


if __name__ == "__main__":
    print(f"H5 to convert path: {H5_TO_CONVERT_PATH}")
    print("======================")
    print("Verifying h5 contents:")
    with h5py.File(H5_TO_CONVERT_PATH, "r") as f:
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

    # Example: extract X_test / y_test into data/datasets/<h5basename>_extracted
    base_out = os.path.join(os.path.dirname(H5_TO_CONVERT_PATH), os.path.basename(H5_TO_CONVERT_PATH).replace('.h5','') + '_extracted')
    print('Default extraction output base:', base_out)
    summary = extract_images_from_h5(H5_TO_CONVERT_PATH, base_out, group_keys=['val', 'train'], overwrite=False, verbose=True)
    print('Extraction summary:', summary)