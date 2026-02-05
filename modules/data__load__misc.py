import h5py
import numpy as np


def load_data_and_labels(file_path, info, occlusion_dataset=False):
    class_names = None
    with h5py.File(file_path, 'r') as f:
        if occlusion_dataset and info == 'train':
            # Dataset looks like this:
            #   X_train.shape: (144469, 128, 128, 3)
            #   X_train dtype: uint8
            #   X_val.shape: (35736, 128, 128, 3)
            #   X_val dtype: uint8
            #   mismatch_train.shape: (144469,)
            #   mismatch_train: [0 0 0 ... 0 0 0]
            #   mismatch_val.shape: (35736,)
            #   mismatch_val: [0 0 0 ... 0 0 0]
            #   occ_train.shape: (144469,)
            #   occ_train: [0 0 0 ... 6 6 6]
            #   occ_val.shape: (35736,)
            #   occ_val: [0 0 0 ... 6 6 6]
            #   pos_or_neg_train.shape: (144469,)
            #   pos_or_neg_train: [1 1 1 ... 1 1 1]
            #   pos_or_neg_val.shape: (35736,)
            #   pos_or_neg_val: [1 1 1 ... 1 1 1]
            #   y_train.shape: (144469,)
            #   y_train: [0 0 0 ... 6 6 6]
            #   y_val.shape: (35736,)
            #   y_val: [0 0 0 ... 6 6 6]
            X_train = np.array(f['X_train'])
            y_train = np.array(f['y_train'])
            X_train_original_hashes = np.array(f['original_hash_train'])
            occ_train = np.array(f['occ_train'])
            mismatch_train = np.array(f['mismatch_train'])
            pos_or_neg_train = np.array(f['pos_or_neg_train'])

            X_val = np.array(f['X_val'])
            y_val = np.array(f['y_val'])
            X_val_original_hashes = np.array(f['original_hash_val'])
            occ_val = np.array(f['occ_val'])
            mismatch_val = np.array(f['mismatch_val'])
            pos_or_neg_val = np.array(f['pos_or_neg_val'])

            class_names = [name.decode('utf-8') for name in f['class_names']]

            return X_train, y_train, X_val, y_val, class_names, X_train_original_hashes, X_val_original_hashes, occ_train, mismatch_train, pos_or_neg_train, occ_val, mismatch_val, pos_or_neg_val

        elif occlusion_dataset and info == 'test':
            raise ValueError("Occlusion dataset should not have test data, but info is 'test'. This does not mean that the test set can't be occluded, just that this flag should only be used for loading the training and validation sets")
        
        elif info == 'train':
            # Dataset looks like this:
            #   X_train.shape: (21332, 128, 128, 3)
            #   X_train dtype: uint8
            #   X_val.shape: (5273, 128, 128, 3)
            #   X_val dtype: uint8
            #   class_names.shape: (7,)
            #   class_names: [b'ANGRY' b'DISGUST' b'FEAR' b'HAPPY' b'NEUTRAL' b'SAD' b'SURPRISE']
            #   y_train.shape: (21332,)
            #   y_train: [0 0 0 ... 6 6 6]
            #   y_val.shape: (5273,)
            #   y_val: [0 0 0 ... 6 6 6]
            X_train = np.array(f['X_train'])
            y_train = np.array(f['y_train'])
            X_val = np.array(f['X_val'])
            y_val = np.array(f['y_val'])
            class_names = [name.decode('utf-8') for name in f['class_names']]
            
            return X_train, y_train, X_val, y_val, class_names
        elif info == 'test':
            # Dataset looks like this:
            #   X_test.shape: (350, 128, 128, 3)
            #   X_test dtype: uint8
            #   class_names.shape: (7,)
            #   class_names: [b'ANGRY' b'DISGUST' b'FEAR' b'HAPPY' b'NEUTRAL' b'SAD' b'SURPRISE']
            #   paths.shape: (350,)
            #   paths (first and last five): [b'.\\data\\datasets\\occluded_test_set\\bosphorus_test_HQ\\ANGRY\\bosphorus_bs001_ANGRY_30__masked-negative-DISGUST_mismatch.png' ... b'.\\data\\datasets\\occluded_test_set\\bosphorus_test_HQ\\SURPRISE\\bosphorus_bs104_SURPRISE_7__masked-positive-SURPRISE_match.png']
            #   y_test.shape: (350,)
            #   y_test: [0 0 ... 6 6]

            X_test = np.array(f['X_test'])
            y_test = np.array(f['y_test'])
            class_names = [name.decode('utf-8') for name in f['class_names']]
            
            return X_test, y_test, class_names
        else:
            raise ValueError(f"Info must be 'train' or 'test', but is '{info}'")
        

def remove_indices_from_data(X_data, y_data, paths_data, indices_to_remove):
    indices_to_remove = sorted(indices_to_remove)

    X_data = np.delete(X_data, indices_to_remove, axis=0)
    y_data = np.delete(y_data, indices_to_remove, axis=0)
    paths_data = np.delete(paths_data, indices_to_remove, axis=0) if paths_data is not None else None
    return X_data, y_data, paths_data