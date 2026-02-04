import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.data__load__misc import load_data_and_labels
from modules.config import ORIGINAL_TRAIN_VAL_SET_H5_PATH
from modules.misc import StatsTracker



def calculate_expected_size(original_size, class_names, emotion_names):
    nof_neutral_images = sum(1 for name in emotion_names if name.lower() == 'neutral')
    nof_occlusion_types_by_emotion = len(class_names) - 1   # -1 for excluding NEUTRAL, should be 6
    nof_occlusion_types_by_posneg = 2                       # positive and negative,    should be 2            - original_size to account for matching occlusions doing only positive and not negative
    expected_size = original_size * nof_occlusion_types_by_emotion * nof_occlusion_types_by_posneg - original_size + nof_neutral_images # + nof_neutral_images to account for neutral images not possibly having matching occlusions
    return expected_size 



if __name__ == "__main__":
    x_train, y_train, x_val, y_val, class_names, train_paths, val_paths = load_data_and_labels(ORIGINAL_TRAIN_VAL_SET_H5_PATH, 'train')

    emotion_names_train =   [class_names[label] for label in y_train]
    emotion_names_val =     [class_names[label] for label in y_val]

    # original_train_size = len(x_train)
    # original_val_size = len(x_val)

    # print(f"Original training set size: {original_train_size} samples")
    # print(f"Original validation set size: {original_val_size} samples")
    print(f"class_names: {class_names}")

    # print(f"I expect the counts to be: ")
    # expected_train_size = calculate_expected_size(original_train_size, class_names, emotion_names_train)
    # expected_val_size = calculate_expected_size(original_val_size, class_names, emotion_names_val)
    # print(f"  Training set: {expected_train_size} samples")
    # print(f"  Validation set: {expected_val_size} samples")

    # print(f"Calculating actual counts considering occlusions...")

    for emotion_names, split in [(emotion_names_train, 'train'), (emotion_names_val, 'val')]:
        original_size = len(emotion_names)

        print(f"Original {split.capitalize()} set size: {original_size} samples")
        print("I expect the counts to be: ")
        expected_size = calculate_expected_size(original_size, class_names, emotion_names)
        print(f"  {split.capitalize()} set: {expected_size} samples")

        count = 0
        for mismatching_emotion in class_names:
            # We apply matching or mismatching occlusions (based on whether the occluding emotion is the same or different from the gt), but NEUTRAL has no landmarks
            if mismatching_emotion.lower() == 'neutral':
                continue  # skip neutral

            for ground_truth in emotion_names:
                # We explore the whole dataset and apply both negative and positive mismatches, with one exception:
                #   - if the ground truth is the same as the emotion (matching emotions), we can only apply a matching occlusion
                if ground_truth.lower() == mismatching_emotion.lower():
                    count += 1
                else:
                    count += 2

        print(f"Total size of {split} set considering occlusions: {count} samples")