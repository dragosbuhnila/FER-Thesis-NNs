import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PIL import Image
import h5py

from modules.config import OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_RESIZED_PATH



def load_extracted_folder_images_filenames(folder_path, extract_idx_function=None):
    """Load all images from the extracted folder."""
    i = 0
    images = {}
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):  # Ensure consistent order
            file_path = os.path.join(root, file)
            
            if extract_idx_function is not None:
                idx = extract_idx_function(file_path)
            else:                
                idx = i  # Default to sequential index if no function provided

            images[idx] = file_path
            i += 1
    return images



IMAGES_FOLDER_PATH = OCCLUDED_TEST_SET_RESIZED_PATH



if __name__ == '__main__':
    # e.g. bosphorus_bs001_ANGRY_30__masked-negative-DISGUST_mismatch.png
    extracted_images_paths = load_extracted_folder_images_filenames(IMAGES_FOLDER_PATH)

    if len(extracted_images_paths) > 999:
        raise RuntimeError(f"Too many images ({len(extracted_images_paths)}) to rename with 3 digits. Please adjust the code to use more digits if needed.")

    for idx, path in extracted_images_paths.items():
        dir_name, filename = os.path.split(path)
        new_filename = f'image_{idx:03d}__' + filename
        new_path = os.path.join(dir_name, new_filename)
        os.rename(path, new_path)
        print(f'Renamed {path} to {new_path}')