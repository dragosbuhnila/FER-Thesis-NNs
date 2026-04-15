import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

from typing import List, Optional, Tuple
import h5py
import re
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from modules.visualize import plot_image
from modules.config import CANONICAL_FACES_CUT, CANONICAL_FACES_CUT_OCCLUDED, EMOTIONS
from modules.data__load import load_online_data_generators
from modules.misc import create_occludedtrainvalset_filename_from_info, StatsTracker


# This is needed only if you don't have an h5 already
ORIGINAL_IMAGES_FOLDER_PATH = CANONICAL_FACES_CUT
ORIGINAL_IMAGES_H5_PATH = os.path.join(ORIGINAL_IMAGES_FOLDER_PATH, f"{os.path.basename(ORIGINAL_IMAGES_FOLDER_PATH)}.h5")
OCCLUDED_IMAGES_PATH = CANONICAL_FACES_CUT_OCCLUDED

# This one is the general settings for trainval and test h5 sets. You may set test to None if you don't have it or need it.
TRAINVAL_PATH = ORIGINAL_IMAGES_H5_PATH
TEST_SET_PATH = None  # not needed for this script, as we only occlude the trainval set for offline dataset generation



parser = argparse.ArgumentParser(description='Generate occluded dataset offline and save to HDF5')
parser.add_argument("-m", "--mismatch", type=str,   required=True, default=False, help="Mismatching emotion.")
parser.add_argument("-p", "--pos_or_neg", type=str, required=False, default=None, help="If mismatch is a specific emotion, whether it is 'positive' or 'negative'.")
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
print(f"    EMOTIONS: {EMOTIONS}")
print(f"    OCCLUDED_IMAGES_PATH: {OCCLUDED_IMAGES_PATH}")
print("ARGS:")
print(f"    Mismatch: {args.mismatch}")
print(f"    Pos or Neg: {args.pos_or_neg}")
print(f"    Batch Size: {args.batch_size}")
print(f"    Show Images: {args.show_images}")
print(f"    Don't Parallelize Loading Landmarks: {args.dont_parallelize_loading_landmarks}")
print("===============================================================================")

            

def save_occluded_dataset_offline(data_generator, generator_name, save_folder_path, specific_mismatch, positive_or_negative, ignore_stats_tracker=False):
    if not ignore_stats_tracker:
        stats_tracker = StatsTracker(EMOTIONS, generator_name, specific_mismatch, positive_or_negative)

    for batch_x, batch_y, batch_x_hashes in tqdm(data_generator, total=len(data_generator), desc=f"Saving occluded images for {generator_name} set"):
        for image, label, img_hash in zip(batch_x, batch_y, batch_x_hashes):
            label_name = EMOTIONS[np.argmax(label)].lower()
            mismatching = "matching" if (label_name == specific_mismatch) else "mismatching"
            
            # Updated filename format
            filename = f"{label_name.upper()}_occ_with_{specific_mismatch.upper()}_{'POS' if positive_or_negative == 'positive' else 'NEG'}.png"
            filepath = os.path.join(save_folder_path, filename)
            os.makedirs(save_folder_path, exist_ok=True)

            if args.show_images:
                plot_image(image, title=f"{filename}")

            if not ignore_stats_tracker:
                stats_tracker.increase_processed()
                stats_tracker.update_from_filename(filename)

            if mismatching == "matching" and positive_or_negative == "negative":
                # skip saving matching negatives
                if not ignore_stats_tracker:
                    stats_tracker.increase_skipped()
                continue

            # Save image
            try:
                # check if image is already saved for some reason
                if os.path.exists(filepath):
                    print(f"[ERROR] Image already exists: {filepath}")
                    continue
                Image.fromarray(image).save(filepath)
                if not ignore_stats_tracker:
                    stats_tracker.increase_saved()
            except Exception as e:
                print(f"[ERROR] Error while saving image {filepath}: {e}")

    if not ignore_stats_tracker:
        stats_tracker.check_consistency()
        print(str(stats_tracker))
    
def create_h5_from_pngs(images_dir: str,
                        h5_path: str,
                        class_order: Optional[List[str]] = None,
                        resize_to: Optional[Tuple[int,int]] = None,
                        compress: bool = True) -> dict:
    """
    Read only top-level PNG files from `images_dir`. Filenames must contain
    the emotion token in ALL CAPS (e.g. "11872 ANGRY.png"). Use that token
    as the ground-truth label (mapped by `class_order`).

    - images_dir: folder containing the 7 PNG files (no subfolders).
    - h5_path: output .h5 file path.
    - class_order: optional list like ["ANGRY","DISGUST","FEAR","HAPPY","NEUTRAL","SAD","SURPRISE"].
                   If None, the common 7-emotion order is used.
    - resize_to: (w,h) to resize all images to the same size. If None, all PNGs must already share the same shape.
    - compress: use gzip compression for the image dataset.
    Returns: dict with keys `n_images` and `class_names`.
    """
    if class_order is None:
        class_order = ["ANGRY","DISGUST","FEAR","HAPPY","NEUTRAL","SAD","SURPRISE"]

    png_files = sorted(f for f in os.listdir(images_dir)
                       if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith('.png'))

    if not png_files:
        raise ValueError(f"No PNG files found in {images_dir}")

    X_list = []
    y_list = []
    paths = []
    emotion_re = re.compile(r'([A-Z]+)(?=\.png$)')  # matches final ALL-CAPS token before .png

    first_shape = None
    for fname in png_files:
        match = emotion_re.search(fname)
        if not match:
            # skip files that don't match the EMOTION pattern
            continue
        emotion = match.group(1)
        try:
            label = class_order.index(emotion)
        except ValueError:
            raise ValueError(f"Emotion '{emotion}' in file '{fname}' not found in class_order: {class_order}")

        img_path = os.path.join(images_dir, fname)
        img = Image.open(img_path).convert('RGB')
        if resize_to:
            img = img.resize(resize_to)
        arr = np.array(img)

        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            raise ValueError(f"Image shapes differ (expected {first_shape}, got {arr.shape}). "
                             "Pass `resize_to` to normalize shapes.")

        X_list.append(arr)
        y_list.append(label)
        paths.append(img_path)

    if not X_list:
        raise ValueError("No valid labeled PNGs found (check filenames for an ALL-CAPS emotion token).")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)

    with h5py.File(h5_path, "w") as f:
        if compress:
            f.create_dataset("X", data=X, compression="gzip")
        else:
            f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        f.create_dataset("class_names", data=np.array(class_order).astype('S'))
        f.create_dataset("paths", data=np.array(paths).astype('S'))

    return {"n_images": X.shape[0], "class_names": class_order}

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch ANGRY --pos_or_neg positive --batch_size 16 -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch ANGRY --pos_or_neg negative --batch_size 16 -d

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch DISGUST --pos_or_neg positive --batch_size 16  -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch DISGUST --pos_or_neg negative --batch_size 16  -d

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch FEAR --pos_or_neg positive --batch_size 16  -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch FEAR --pos_or_neg negative --batch_size 16  -d

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch HAPPY --pos_or_neg positive --batch_size 16  -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch HAPPY --pos_or_neg negative --batch_size 16  -d

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch SAD --pos_or_neg positive --batch_size 16  -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch SAD --pos_or_neg negative --batch_size 16  -d

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch SURPRISE --pos_or_neg positive --batch_size 16  -d
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/unsorted_tools/occlude_images.py" --mismatch SURPRISE --pos_or_neg negative --batch_size 16  -d
if __name__ == "__main__":
    # make an h5 dataset out of the input
    create_h5_from_pngs(ORIGINAL_IMAGES_FOLDER_PATH, ORIGINAL_IMAGES_H5_PATH)

    data_generator, _, _, _ = load_online_data_generators(
                                                                # Paths ---------------------------------------------------
                                                                trainval_path=TRAINVAL_PATH, 
                                                                test_path=TEST_SET_PATH,
                                                                # We need to occlude all the images for offline dataset ---
                                                                training_occlusion_probability=1.0,       
                                                                validation_occlusion_probability=1.0,
                                                                # Mismatch settings ---------------------------------------
                                                                mismatch=args.mismatch,
                                                                pos_or_neg=args.pos_or_neg,
                                                                # Command line args for working ---------------------------
                                                                batch_size=args.batch_size,
                                                                parallelize=not args.dont_parallelize_loading_landmarks,
                                                                # Hardcoded -----------------------------------------------
                                                                masking_function_name='lines',
                                                                use_label_smoothing=True,
                                                                dont_augment=True,
                                                                dont_rebalance_trainval=True,
                                                                yield_hashes=True,
                                                                remove_dupes=False,
                                                                run_detection=True,
                                                            )

    specific_mismatch = args.mismatch.lower()
    if args.pos_or_neg is not None:
        positive_or_negative = args.pos_or_neg.lower()
    else:
        raise ValueError("If mismatch is a specific emotion, pos_or_neg must be provided.")
    
    save_occluded_dataset_offline(data_generator, "train", OCCLUDED_IMAGES_PATH, specific_mismatch, positive_or_negative, ignore_stats_tracker=True)
    