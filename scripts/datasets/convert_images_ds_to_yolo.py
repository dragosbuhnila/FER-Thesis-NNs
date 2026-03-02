import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tqdm import tqdm

from modules.config import OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,\
                            EMOTIONS



if __name__ == "__main__":
    print(f"TRAIN_SET_IMAGES_PATH: {OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH}")
    print(f"VAL_SET_IMAGES_PATH: {OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH}")

    # original file structure:
    # - occluded_and_original_trainval_set
    #   - images
    #     - train
    #       - gt-angry_occ-angry
    #           - 0a4f9d85f32c5c0b00f17a978161cbcd_gt-angry_occ-angry_matching_positive.png
    #       - gt-angry_occ-disgust
    #       - ...
    #       - gt-happy_occ-happy
    #       - ...
    #     - val
    #       - gt-angry_occ-angry
    #       - ...

    # desired file structure (yolo compatible):
    # - occluded_and_original_trainval_set
    #   - images
    #     - train
    #      - ANGRY
    #      - DISGUST
    #      - ...

    # Go through all images, extract gt, make it caps, and that will be the class, so save the path of each image in a dictionary and thus move them to the new folder structure
    # for split_path in [OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH]:
    for split_path in [OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH]:
    # for split_path in [OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH]:
        new_split_path = os.path.join(os.path.dirname(split_path), os.path.basename(split_path) + "_new")
        # Find the files and compute the corresponding new paths
        old_paths_to_new_paths = {}
        images_count = 0
        all_files = []
        all_dirs = set()

        # Collect all files first
        for root, dirs, files in os.walk(split_path):
            for directory in dirs:
                all_dirs.add(directory)
            for file in files:
                if file.endswith(".png"):
                    all_files.append((root, file))

        # Check if process has not already been done
        for emotion in EMOTIONS:
            if emotion in all_dirs:
                print(f"[WARNING] Found directory named '{emotion}' in {split_path}. This might indicate that the conversion process has already been done. Please check the directory structure before proceeding.")
                exit(1)

        # Use tqdm to iterate over the collected files
        for root, file in tqdm(all_files, desc=f"Processing {split_path}"):
            images_count += 1
            # Extract gt from filename
            gt_part = file.split("_")[1]  # e.g. "gt-angry"
            gt_label = gt_part.split("-")[1].upper()  # e.g. "ANGRY"
            # Create new folder path
            new_folder_path = os.path.join(new_split_path, gt_label)
            # Move file to new folder
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_folder_path, file)
            old_paths_to_new_paths[old_file_path] = new_file_path

        if images_count == 0:
            print(f"[WARNING] No images found in {split_path}. Please check the path and ensure it contains .png files.")
        if len(old_paths_to_new_paths) != images_count:
            raise ValueError(f"[WARNING] Found {images_count} images but computed {len(old_paths_to_new_paths)} new paths. There might be an issue with the filename parsing logic.")

        # Now move the files
        for old_path, new_path in tqdm(old_paths_to_new_paths.items(), desc=f"Moving files for {split_path}"):
            # old_path_rel = os.path.relpath(old_path, os.path.join(split_path, ".."))
            # new_path_rel = os.path.relpath(new_path, os.path.join(split_path, ".."))

            # print(f"Moving/Renaming the following files (old above new below):")
            # print(f"\t{old_path_rel}")
            # print(f"\t{new_path_rel}")

            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)

