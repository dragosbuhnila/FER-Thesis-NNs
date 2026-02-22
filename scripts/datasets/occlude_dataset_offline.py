import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from modules.visualize import plot_image
from modules.config import EMOTIONS, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_SET_IMAGES_PATH, OCCLUDED_VAL_SET_IMAGES_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
from modules.data__load import load_online_data_generators
from modules.misc import create_occludedtrainvalset_filename_from_info, hash_image, StatsTracker



# ======================
# X_train.shape: (21332, 128, 128, 3)
# X_train dtype: uint8
# X_val.shape: (5273, 128, 128, 3)
# X_val dtype: uint8
# class_names.shape: (7,)
# class_names: [b'ANGRY' b'DISGUST' b'FEAR' b'HAPPY' b'NEUTRAL' b'SAD' b'SURPRISE']
# y_train.shape: (21332,)
# y_train: [0 0 0 ... 6 6 6]
# y_val.shape: (5273,)
# y_val: [0 0 0 ... 6 6 6]
# ======================



TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
TRAIN_OCCLUDED_IMAGES_PATH = OCCLUDED_TRAIN_SET_IMAGES_PATH
VAL_OCCLUDED_IMAGES_PATH = OCCLUDED_VAL_SET_IMAGES_PATH
# I don't need test set for this script but to reuse the loading function i'll add it
TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH



parser = argparse.ArgumentParser(description='Generate occluded dataset offline and save to HDF5')
parser.add_argument("-m", "--mismatch", type=str,   required=True, default=False, help="Whether to use mismatched occlusions (boolean).")
parser.add_argument("-p", "--pos_or_neg", type=str, required=False, default=None, help="If mismatch is a specific emotion, whether it is 'positive' or 'negative'.")
parser.add_argument("-s", "--small_subset", action='store_true', help="If set, use a small subset of the data for quick testing.")
parser.add_argument("-b", "--batch_size", type=int, required=False, default=32, help="Batch size for data generator.")
parser.add_argument("-d", "--dont_parallelize_loading_landmarks", action='store_true', help="If set, do not parallelize loading landmarks.")
parser.add_argument("--show_images", action='store_true', help="If set, display images as they are generated.")
args = parser.parse_args()


if args.mismatch in args.mismatch.lower() == 'uniform':
    raise ValueError("You shouldn't be using 'uniform' mismatch for offline occluded dataset generation")
elif args.mismatch.upper() in EMOTIONS:
    if args.mismatch.upper() == 'NEUTRAL':
        raise ValueError("NEUTRAL emotion cannot be used for mismatch, as it is not allowed to mismatch to NEUTRAL.") 
    if args.pos_or_neg is None:
        raise ValueError("If mismatch is a specific emotion, pos_or_neg must be provided as 'positive' or 'negative'.")
    if args.pos_or_neg.lower() not in ['positive', 'negative']:
        raise ValueError("pos_or_neg must be either 'positive' or 'negative'.")
else:
    raise ValueError(f"Invalid mismatch value: {args.mismatch}. Must be one of {EMOTIONS}.")


print("================================== SETTINGS ==================================")
print(f"MACROS: ")
print(f"    TRAINVAL_SET_PATH: {TRAINVAL_SET_PATH}")
print(f"    TRAIN_OCCLUDED_IMAGES_PATH: {TRAIN_OCCLUDED_IMAGES_PATH}")
print(f"    VAL_OCCLUDED_IMAGES_PATH: {VAL_OCCLUDED_IMAGES_PATH}")
print(f"    TEST_SET_PATH: {TEST_SET_PATH}")
print("ARGS:")
print(f"    Mismatch: {args.mismatch}")
print(f"    Pos or Neg: {args.pos_or_neg}")
print(f"    Small Subset: {args.small_subset}")
print(f"    Batch Size: {args.batch_size}")
print(f"    Show Images: {args.show_images}")
print(f"    Don't Parallelize Loading Landmarks: {args.dont_parallelize_loading_landmarks}")
print("===============================================================================")

            

def save_occluded_dataset_offline(data_generator, generator_name, save_folder_path, specific_mismatch, positive_or_negative):
    stats_tracker = StatsTracker(EMOTIONS, generator_name, specific_mismatch, positive_or_negative)

    for batch_x, batch_y, batch_x_hashes in tqdm(data_generator, total=len(data_generator), desc=f"Saving occluded images for {generator_name} set"):
        for image, label, img_hash in zip(batch_x, batch_y, batch_x_hashes):
            label_name = EMOTIONS[np.argmax(label)].lower()
            mismatching = "matching" if (label_name == specific_mismatch) else "mismatching"
            
            filename = create_occludedtrainvalset_filename_from_info(img_hash, label_name, specific_mismatch, mismatching, positive_or_negative)
            
            sub_folder = f"gt-{label_name}_occ-{specific_mismatch}"
            filepath = os.path.join(save_folder_path, sub_folder, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if args.show_images:
                folder_name_short = os.path.basename(os.path.dirname(filepath))
                plot_image(image, title=f"{folder_name_short} / {filename}")

            stats_tracker.increase_processed()
            stats_tracker.update_from_filename(filename)

            if mismatching == "matching" and positive_or_negative == "negative":
                # skip saving matching negatives
                stats_tracker.increase_skipped()
                continue

            # Save image
            try:
                # check if image is already saved for some reason
                if os.path.exists(filepath):
                    print(f"[ERROR] Image already exists: {filepath}")
                    continue
                Image.fromarray(image).save(filepath)
                stats_tracker.increase_saved()
            except Exception as e:
                print(f"[ERROR] Error while saving image {filepath}: {e}")

    stats_tracker.check_consistency()
    print(str(stats_tracker))
    


# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/occlude_dataset_offline.py" --mismatch ANGRY --pos_or_neg positive --small_subset --batch_size 16 --show_images -d
if __name__ == "__main__":

    train_generator, val_generator, _, _ = load_online_data_generators(
                                                                # Paths ---------------------------------------------------
                                                                trainval_path=TRAINVAL_SET_PATH, 
                                                                test_path=TEST_SET_PATH,
                                                                # We need to occlude all the images for offline dataset ---
                                                                training_occlusion_probability=1.0,       
                                                                validation_occlusion_probability=1.0,
                                                                # Mismatch settings ---------------------------------------
                                                                mismatch=args.mismatch,
                                                                pos_or_neg=args.pos_or_neg,
                                                                # Command line args for working ---------------------------
                                                                small_subset=args.small_subset,
                                                                batch_size=args.batch_size,
                                                                parallelize=not args.dont_parallelize_loading_landmarks,
                                                                # Hardcoded -----------------------------------------------
                                                                masking_function_name='lines',
                                                                use_label_smoothing=True,
                                                                dont_augment=True,
                                                                dont_rebalance_trainval=True,
                                                                yield_hashes=True,
                                                            )

    specific_mismatch = args.mismatch.lower()
    if args.pos_or_neg is not None:
        positive_or_negative = args.pos_or_neg.lower()
    else:
        raise ValueError("If mismatch is a specific emotion, pos_or_neg must be provided.")
    
    save_occluded_dataset_offline(train_generator, "train", TRAIN_OCCLUDED_IMAGES_PATH, specific_mismatch, positive_or_negative)
    save_occluded_dataset_offline(val_generator, "validation", VAL_OCCLUDED_IMAGES_PATH, specific_mismatch, positive_or_negative)

    

    # Later, convert saved images to HDF5