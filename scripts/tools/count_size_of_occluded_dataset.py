import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tqdm import tqdm

from modules.data__load__misc import load_data_and_labels
from modules.config import ORIGINAL_TRAIN_VAL_SET_H5_PATH

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, class_names, train_paths, val_paths = load_data_and_labels(ORIGINAL_TRAIN_VAL_SET_H5_PATH, 'train')

    emotion_names_train =   [class_names[label] for label in y_train]
    emotion_names_val =     [class_names[label] for label in y_val]


    for emotion_names, split in [(emotion_names_train, 'train'), (emotion_names_val, 'val')]:
        count = 0
        for emotion in class_names:
            if emotion.lower() == 'neutral':
                continue  # skip neutral

            for ground_truth in emotion_names:
                if ground_truth.lower() == emotion.lower():
                    count += 1
                else:
                    count += 2

        print(f"Total size of {split} set considering occlusions: {count} samples")