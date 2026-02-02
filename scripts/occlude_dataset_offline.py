import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from modules.visualize import plot_image
from modules.config import EMOTIONS, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_SET_IMAGES_PATH, OCCLUDED_VAL_SET_IMAGES_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
from modules.data__load import load_data_generators
from modules.misc import hash_image;



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



class StatsTracker:
    def __init__(self, emotions, generator_name, specific_mismatch, positive_or_negative):
        self.generator_name = generator_name
        self.specific_mismatch = specific_mismatch
        self.positive_or_negative = positive_or_negative

        # Initialize the stats dictionary
        self.stats = {
            # Processing
            'processed_images': 0,
            'skipped_images': 0,
            'saved_images': 0,
            # Types of images
            'matching_images': 0,
            'mismatching_images': 0,
            'positive_images': 0,
            'negative_images': 0,
        }

        # Add emotion-specific stats
        for emotion in emotions:
            self.stats[f'gt-{emotion.lower()}_images'] = 0
            self.stats[f'occ-{emotion.lower()}_images'] = 0

    def __str__(self):
        stats_str =  "=== Stats Tracker ===\n"
        stats_str += f"Generator Name: {self.generator_name}\n"
        stats_str += f"Specific Mismatch: {self.specific_mismatch}\n"
        stats_str += f"Positive or Negative: {self.positive_or_negative}\n"
        for key, value in self.stats.items():
            stats_str += f"{key}: {value}\n"
        stats_str += "=====================\n"
        return stats_str

    def update_from_filename(self, filename):
        # Parse the filename
        _, gt_emotion, occ_emotion, mismatching, pos_or_neg = filename[:-4].split('_')

        # Update matching/mismatching stats
        if mismatching == "matching":
            self.stats['matching_images'] += 1
        elif mismatching == "mismatching":
            self.stats['mismatching_images'] += 1
        else:
            raise ValueError(f"Filename parsing error for mismatching. Expected 'matching' or 'mismatching', got {mismatching}.")

        # Update positive/negative stats
        if pos_or_neg == "positive":
            self.stats['positive_images'] += 1
        elif pos_or_neg == "negative":
            self.stats['negative_images'] += 1
        else:
            raise ValueError(f"Filename parsing error for pos_or_neg. Expected 'positive' or 'negative', got {pos_or_neg}.")

        # Update emotion-specific stats
        self.stats[f'{gt_emotion}_images'] += 1
        self.stats[f'{occ_emotion}_images'] += 1

    def check_consistency(self):
        # Check if processed images match skipped + saved images
        if self.stats['processed_images'] != (self.stats['skipped_images'] + self.stats['saved_images']):
            print(f"[WARNING] Processed images ({self.stats['processed_images']}) != "
                  f"Skipped images ({self.stats['skipped_images']}) + Saved images ({self.stats['saved_images']})")
            print(str(self))
            
    def increase_processed(self):
        self.stats['processed_images'] += 1

    def increase_skipped(self):
        self.stats['skipped_images'] += 1

    def increase_saved(self):
        self.stats['saved_images'] += 1

            

def save_occluded_dataset_offline(data_generator, generator_name, save_folder_path, specific_mismatch, positive_or_negative):
    stats_tracker = StatsTracker(EMOTIONS, generator_name, specific_mismatch, positive_or_negative)

    for batch_x, batch_y in tqdm(data_generator, total=len(data_generator), desc=f"Saving occluded images for {generator_name} set"):
        for image, label in zip(batch_x, batch_y):
            img_hash = hash_image(image)
            label_name = EMOTIONS[np.argmax(label)].lower()
            mismatching = "matching" if (label_name == specific_mismatch) else "mismatching"
            
            filename = f"{img_hash}_gt-{label_name}_occ-{specific_mismatch}_{mismatching}_{positive_or_negative}.png"
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
                Image.fromarray(image).save(filepath)
                stats_tracker.increase_saved()
            except Exception as e:
                print(f"Error saving image {filepath}: {e}")

    stats_tracker.check_consistency()
    print(str(stats_tracker))
    


# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/occlude_dataset_offline.py" --mismatch ANGRY --pos_or_neg positive --small_subset --batch_size 16 --show_images -d
if __name__ == "__main__":

    train_generator, val_generator, _, _ = load_data_generators(
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
                                                            )

    specific_mismatch = args.mismatch.lower()
    if args.pos_or_neg is not None:
        positive_or_negative = args.pos_or_neg.lower()
    else:
        raise ValueError("If mismatch is a specific emotion, pos_or_neg must be provided.")
    
    save_occluded_dataset_offline(train_generator, "train", TRAIN_OCCLUDED_IMAGES_PATH, specific_mismatch, positive_or_negative)
    save_occluded_dataset_offline(val_generator, "validation", VAL_OCCLUDED_IMAGES_PATH, specific_mismatch, positive_or_negative)

    

    # Later, convert saved images to HDF5