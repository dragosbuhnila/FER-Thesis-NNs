import h5py
import numpy as np


def load_data_and_labels(file_path, info):
    class_names = None
    with h5py.File(file_path, 'r') as f:
        if info == 'train':
            X_train = np.array(f['X_train'])
            y_train = np.array(f['y_train'])
            X_val = np.array(f['X_val'])
            y_val = np.array(f['y_val'])
            class_names = [name.decode('utf-8') for name in f['class_names']]

            if 'paths' in f:
                # Se 'paths' è un dataset di stringhe a lunghezza variabile
                # con h5py.string_dtype, possiamo leggerlo direttamente:
                train_paths_data = f['train_paths'][...]  # np array di stringhe
                val_paths_data = f['val_paths'][...]  # np array di stringhe
            else:
                train_paths_data = None
                val_paths_data = None
            
            return X_train, y_train, X_val, y_val, class_names, train_paths_data, val_paths_data
        elif info == 'test':
            X_test = np.array(f['X_test'])
            y_test = np.array(f['y_test'])
            class_names = [name.decode('utf-8') for name in f['class_names']]
            
            if 'paths' in f:
                # Se 'paths' è un dataset di stringhe a lunghezza variabile
                # con h5py.string_dtype, possiamo leggerlo direttamente:
                paths_data = f['paths'][...]  # np array di stringhe
            else:
                paths_data = None
            return X_test, y_test, class_names, paths_data
        else:
            raise ValueError(f"Info must be 'train' or 'test', but is '{info}'")


def remove_indices_from_data(X_data, y_data, paths_data, indices_to_remove):
    indices_to_remove = sorted(indices_to_remove)

    X_data = np.delete(X_data, indices_to_remove, axis=0)
    y_data = np.delete(y_data, indices_to_remove, axis=0)
    paths_data = np.delete(paths_data, indices_to_remove, axis=0) if paths_data is not None else None
    return X_data, y_data, paths_data