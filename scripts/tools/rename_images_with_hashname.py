import os; import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PIL import Image
from tqdm import tqdm

from modules.misc import hash_image
from modules.config import LANDMARK_COORDINATES_FOLDER_PATH, ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH, ADELE_TEST_SET_IMAGES_PATH



IMAGES_FOLDERS = [
    ORIGINAL_TRAIN_SET_IMAGES_PATH, 
    ORIGINAL_VAL_SET_IMAGES_PATH,
    ADELE_TEST_SET_IMAGES_PATH,
    ]



if __name__ == "__main__":
    for images_folder in IMAGES_FOLDERS:
        emotion_folders = [ os.path.join(images_folder, d) for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d)) ]

        # 1) Create dictionary mapping image paths to their hash names
        mapping = {}
        all_hash_names = set()
        hashes_in_landmark_coords_folder = {filename.split('.')[0] for filename in os.listdir(LANDMARK_COORDINATES_FOLDER_PATH) if filename.endswith('.npy')}

        # 2) Save hash names to the mapping
        any_non_converted_entry_found = False
        for emotion_folder in emotion_folders:
            for filename in tqdm(os.listdir(emotion_folder)):
                file_path =  os.path.join(emotion_folder, filename)
                try:
                    with Image.open(file_path) as img:
                        img_hash = hash_image(img)

                    new_file_path = os.path.join(emotion_folder, f"{img_hash}.png")
                    mapping[file_path] = new_file_path
                    all_hash_names.add(img_hash)

                    if file_path != new_file_path:
                        any_non_converted_entry_found = True
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        print(f"\nRenaming results for folder: {images_folder}")
        for old_path, hash_path in mapping.items():
            print(f"{old_path:<75}  -->  {hash_path}")

        print(f"Total unique hash names: {len(all_hash_names)}")

        if not any_non_converted_entry_found:
            print("All files are already converted to hash names.")
            continue

        # 3) Check if all hash names are inside the landmark coordinates only for ADELE_TEST_SET_IMAGES_PATH, since I know I already processed it
        if images_folder == ADELE_TEST_SET_IMAGES_PATH:
            missing_found = False
            for hash in all_hash_names:
                if hash not in hashes_in_landmark_coords_folder:
                    print(f"Hash {hash} is not found in landmark coordinates.")
                    missing_found = True
            if not missing_found:
                print("All hash names are found in landmark coordinates folder.")

        # 4) Rename the files
        for old_path, hash_path in tqdm(mapping.items()):
            try:
                os.rename(old_path, hash_path)
            except Exception as e:
                print(f"Error renaming {old_path} to {hash_path}: {e}")