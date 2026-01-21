import os
from joblib import Parallel, delayed
import numpy as np
import h5py
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from modules.config import BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT, EMOTIONS, LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE, LANDMARK_COORDINATES_FOLDER_PATH
from modules.data import CustomBalancedDataGenerator
from modules.landmark_utils import detect_facial_landmarks, load_landmark_coordinates
from modules.misc import hash_image



# def occlude_debug(images, labels, image_landmarks, image_hashes):
#         # images is a tf.Tensor
#         # For now convert it to numpy for easier processing
#         images = images.numpy()

#         # a) Get all the landmarks, in the form of coordinates, for each image
#         #       i.e. for each image every single face point, even unnecessary ones, will be there (they should be cached already)
#         # ____________________________________________________________________________________________
#         # If no landmarks are detected, then occlusion isn't possible for the approach we're currently using (i.e. masking based on AU landmarks),
#         #   so leave the unprocessable image out and reinsert it with no occlusion
#         error_indices = []
#         for i, landmarks_all in enumerate(image_landmarks):
#             if len(landmarks_all) == 0:
#                 error_indices.append(i)

#         # # a1) Prepare unoccludable images to be reinserted later (won't do it in this version as I am already removing the problem images)
#         # unoccludable_images = dict()
#         # # sort error indices in reverse order so that the insertion has correct indices (deleting from the end first)
#         # error_indices.sort(reverse=True)
#         # for i in error_indices:
#         #     # Save the image without occlusion and remove it from the batch
#         #     unoccludable_images[i] = numpy_images[i]
#         #     image_landmarks = np.delete(image_landmarks, i, axis=0) 
#         # # ____________________________________________________________________________________________           
        
#         # b) Get the coordinates relating to just the specific emotion needed, for each image
#         # ____________________________________________________________________________________________
#         emotions = [EMOTIONS[label] for label in labels]
#         list_of_landmark_sets = get_landmark_coordinate_sets_by_emotion__batch(image_landmarks, emotions)
#         # ____________________________________________________________________________________________

#         # c) Apply occlusion based on the landmarks
#         # ____________________________________________________________________________________________
#         occluded = apply_mask_to__batch(images, list_of_landmark_sets, "lines")
#         # ___________________________________________________________________________________________   

#         nop_variable = 0
#         for image, landmarks_emotion, landmarks_emotions_index, hash in zip(occluded, emotions, labels, image_hashes):
#             print(f"Hash for the current image: {hash}")
#             landmarks_that_should_be = load_landmark_coordinates(hash)
#             plot_image(image, title=f"landmarks for emotion {landmarks_emotion} (index {landmarks_emotions_index})\nHash: {hash}")
#             nop_variable += 1


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

def load_test_generator(path, occlusion_probability, masking_function, mismatch, batch_size=64):
    X_test, y_test, class_names, test_paths = load_data_and_labels(path, 'test')

    for emotion in EMOTIONS:
        if emotion not in class_names:
            raise ValueError(f"Class '{emotion}' not found in class names from H5 file.")
    NUM_CLASSES = len(class_names)

    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)
    X_test_hashes = np.array([hash_image(img) for img in X_test])

    data_generator = CustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        x_hashes=X_test_hashes,
        paths_data=test_paths,

        data_inf='test',
        batch_size=batch_size,        
        augmentations={},
        label_smoothing=0,

        masking_function=masking_function,
        occlusion_probability=occlusion_probability,
        mismatch=mismatch,
    )

    return data_generator

def remove_indices_from_data(X_data, y_data, paths_data, indices_to_remove):
    indices_to_remove = sorted(indices_to_remove)

    X_data = np.delete(X_data, indices_to_remove, axis=0)
    y_data = np.delete(y_data, indices_to_remove, axis=0)
    paths_data = np.delete(paths_data, indices_to_remove, axis=0) if paths_data is not None else None
    return X_data, y_data, paths_data

def load_data_generators(train_path, test_path, occlusion_probability, masking_function, use_label_smoothing, mismatch, small_subset=False, run_detection=False, remove_dupes=True, parallelize=True, matching_amount=0.2, batch_size=64, validation_occlusion_probability=0.5):
    # 1) Load training and validation data
    # ____________________________________
    X_train, y_train, X_val, y_val, trainval_class_names, train_paths_data, val_paths_data = load_data_and_labels(train_path, 'train')
    X_test, y_test, test_class_names, test_paths_data = load_data_and_labels(test_path, 'test')

    if remove_dupes:
        for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
            if split == 'X_train':
                X_train, y_train, train_paths_data = remove_indices_from_data(X_train, y_train, train_paths_data, indices)
            elif split == 'X_val':
                X_val, y_val, val_paths_data = remove_indices_from_data(X_val, y_val, val_paths_data, indices)
            elif split == 'X_test':
                X_test, y_test, test_paths_data = remove_indices_from_data(X_test, y_test, test_paths_data, indices)

    if small_subset:
        debug_limit = 100
        X_train = X_train[:debug_limit]
        y_train = y_train[:debug_limit]
        X_val = X_val[:debug_limit]
        y_val = y_val[:debug_limit]
        train_paths_data = train_paths_data[:debug_limit] if train_paths_data is not None else None
        val_paths_data = val_paths_data[:debug_limit] if val_paths_data is not None else None
        X_test = X_test[:debug_limit]
        y_test = y_test[:debug_limit]
        test_paths_data = test_paths_data[:debug_limit] if test_paths_data is not None else None
    
    # 1.b) Hashing
    # ____________________________________
    X_train_hashes = np.array([hash_image(img) for img in X_train])
    X_val_hashes = np.array([hash_image(img) for img in X_val])
    X_test_hashes = np.array([hash_image(img) for img in X_test])
    
    # 1.c) Classes validations
    # ____________________________________
    for emotion in EMOTIONS:
        if emotion not in trainval_class_names:
            raise ValueError(f"Class '{emotion}' not found in training/validation class names from H5 file.")
        if emotion not in test_class_names:
            raise ValueError(f"Class '{emotion}' not found in test class names from H5 file.")
    class_names = test_class_names

    # 1.d) Paths validations
    # ____________________________________
    if train_paths_data is not None and val_paths_data is None:
        raise ValueError(f"Training paths are provided but validation paths are missing, which is weird, so check the dataset at {train_path}.")
    if train_paths_data is None and val_paths_data is not None:
        raise ValueError(f"Validation paths are provided but training paths are missing, which is weird, so check the dataset at {train_path}.")
    
    # 2) Landmarking
    # ____________________________________
    if not run_detection:
        size_of_cached_landmarks = len(os.listdir(LANDMARK_COORDINATES_FOLDER_PATH))
        if size_of_cached_landmarks != LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE:
            raise ValueError(f"The size of the landmark coordinates cache ({size_of_cached_landmarks}) does not match the expected size ({LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE}).\n\
                             This may also mean that the db size has changed.\n\
                             Anyway, you may want to try running ./scripts/tools/find_unlandmarkable.py or download the cache again.")
    # ____________________________________
    if run_detection:
        if parallelize:
            X_train_landmarks = np.array(Parallel(n_jobs=-1)(   delayed(detect_facial_landmarks)(img, img_hash, False, True, True) for img, img_hash in tqdm(zip(X_train, X_train_hashes), desc="Detecting training landmarks")     ))
            X_val_landmarks = np.array(Parallel(n_jobs=-1)(     delayed(detect_facial_landmarks)(img, img_hash, False, True, True) for img, img_hash in tqdm(zip(X_val, X_val_hashes), desc="Detecting validation landmarks")     ))
            X_test_landmarks = np.array(Parallel(n_jobs=-1)(    delayed(detect_facial_landmarks)(img, img_hash, False, True, True) for img, img_hash in tqdm(zip(X_test, X_test_hashes), desc="Detecting test landmarks")          ))
        else:
            X_train_landmarks = np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_train, X_train_hashes)])
            X_val_landmarks =   np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_val, X_val_hashes)])
            X_test_landmarks =  np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_test, X_test_hashes)])
    else:
        if parallelize:
            X_train_landmarks = np.array(Parallel(n_jobs=-1)(   delayed(load_landmark_coordinates)(X_train_hash)    for X_train_hash in tqdm(X_train_hashes, desc="Loading training landmarks")     ))
            X_val_landmarks = np.array(Parallel(n_jobs=-1)(     delayed(load_landmark_coordinates)(X_val_hash)      for X_val_hash in   tqdm(X_val_hashes, desc="Loading validation landmarks")     ))
            X_test_landmarks = np.array(Parallel(n_jobs=-1)(    delayed(load_landmark_coordinates)(X_test_hash)     for X_test_hash in  tqdm(X_test_hashes, desc="Loading test landmarks")          ))
        else:
            X_train_landmarks = np.array([load_landmark_coordinates(X_train_hash) for X_train_hash in tqdm(X_train_hashes, desc="Loading training landmarks")])
            X_val_landmarks =   np.array([load_landmark_coordinates(X_val_hash)   for X_val_hash in   tqdm(X_val_hashes, desc="Loading validation landmarks")])
            X_test_landmarks =  np.array([load_landmark_coordinates(X_test_hash)  for X_test_hash in  tqdm(X_test_hashes, desc="Loading test landmarks")])
    print(f"Landmarks detected for training, validation, and test sets. X_train_landmarks length: {len(X_train_landmarks)}, X_val_landmarks length: {len(X_val_landmarks)}, X_test_landmarks length: {len(X_test_landmarks)}")

    # 2b) Remove 0-length landmark entries (i.e. images where no landmarks were detected)
    #           I filter them here already so that I don't have to handle it at runtime
    # ____________________________________
    def filter_zero_length_landmarks(X_data, y_data, X_hashes, X_landmarks, paths_data=None, name="no_name"):
        # valid_indices = [i for i, landmarks in enumerate(X_landmarks) if len(landmarks) > 0]
        valid_indices = []
        for i, landmarks in enumerate(X_landmarks):
            if len(landmarks) > 0:
                valid_indices.append(i)

        print(f"Found a total of {len(X_landmarks) - len(valid_indices)} invalid indices with zero-length landmarks in the {name} set.")

        X_data_filtered = X_data[valid_indices]
        y_data_filtered = y_data[valid_indices]
        X_hashes_filtered = X_hashes[valid_indices]
        X_landmarks_filtered = X_landmarks[valid_indices]
        if paths_data is not None:
            paths_data_filtered = paths_data[valid_indices]
        
        if paths_data is not None:
            return X_data_filtered, y_data_filtered, X_hashes_filtered, X_landmarks_filtered, paths_data_filtered
        else:
            return X_data_filtered, y_data_filtered, X_hashes_filtered, X_landmarks_filtered, paths_data

    X_train, y_train, X_train_hashes, X_train_landmarks, train_paths_data = filter_zero_length_landmarks(X_train, y_train, X_train_hashes, X_train_landmarks, train_paths_data, name="train")
    X_val, y_val, X_val_hashes, X_val_landmarks, val_paths_data = filter_zero_length_landmarks(X_val, y_val, X_val_hashes, X_val_landmarks, val_paths_data, name="val")
    X_test, y_test, X_test_hashes, X_test_landmarks, test_paths_data = filter_zero_length_landmarks(X_test, y_test, X_test_hashes, X_test_landmarks, test_paths_data, name="test")
    
    train_length = len(X_train); val_length = len(X_val); test_length = len(X_test)
    print(f"After filtering zero-length landmarks: X_train length: {train_length}, X_val length: {val_length}, X_test length: {test_length}. Cumulated length: {train_length + val_length + test_length}")

    # for images, labels, image_landmarks, image_hashes in zip(X_train, y_train, X_train_landmarks, X_train_hashes):
    #     occlude_debug(images, np.argmax(labels, axis=1), image_landmarks, image_hashes, training=True)

    # 3) Compute initial bias
    # ____________________________________
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_probabilities = class_counts / total_samples
    initial_bias = np.log(class_probabilities / (1 - class_probabilities))
    # print("Bias iniziale per ciascuna classe:", initial_bias)

    # 3) Shuffle training and validation data
    # ____________________________________
    # in data.py we shuffle the following:
    #       batch_x, batch_y, batch_x_hashes, batch_x_landmarks, batch_paths = shuffle(batch_x, batch_y, batch_x_hashes, batch_x_landmarks, batch_paths)
    if train_paths_data is not None:
        X_train, y_train, X_train_hashes, X_train_landmarks, train_paths_data = shuffle(X_train, y_train, X_train_hashes, X_train_landmarks, train_paths_data)
        X_val, y_val, X_val_hashes, X_val_landmarks, val_paths_data = shuffle(X_val, y_val, X_val_hashes, X_val_landmarks, val_paths_data)
    else:
        X_train, y_train, X_train_hashes, X_train_landmarks = shuffle(X_train, y_train, X_train_hashes, X_train_landmarks)
        X_val, y_val, X_val_hashes, X_val_landmarks = shuffle(X_val, y_val, X_val_hashes, X_val_landmarks)
    
    # 4) One-hot encoding, augmentations, generators
    # ____________________________________
    NUM_CLASSES = len(class_names)
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)

    # 5)  Augmentations
    # ____________________________________
    # TODO: make them GPUable as right now they can only run on CPU so I have to convert them to numpy arrays
    train_augmentations = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'shear_range': 0.3,
        'horizontal_flip': True,
        'fill_mode': 'wrap',
    }

    # 6) Label smoothing
    # ____________________________________
    if use_label_smoothing:
        label_smoothing_value = 0.05
    else:
        label_smoothing_value = 0.0

    # 7) Create generators
    # ____________________________________
    train_generator = CustomBalancedDataGenerator(
        x_data=X_train, 
        y_data=y_train_one_hot,
        x_hashes=X_train_hashes,
        x_landmarks=X_train_landmarks,
        paths_data=train_paths_data,
        data_inf='train',
        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=label_smoothing_value,
        masking_function=masking_function,
        occlusion_probability=occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
        )
    val_generator = CustomBalancedDataGenerator(
        x_data=X_val,
        y_data=y_val_one_hot,
        x_hashes=X_val_hashes,
        x_landmarks=X_val_landmarks,
        paths_data=val_paths_data,
        data_inf='valid',
        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=0,
        masking_function=masking_function,
        occlusion_probability=validation_occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
        )
    test_generator = CustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        x_hashes=X_test_hashes,
        x_landmarks=X_test_landmarks,
        paths_data=test_paths_data,
        data_inf='test',
        batch_size=batch_size,
        augmentations={},
        label_smoothing=0,
        masking_function=masking_function,
        occlusion_probability=1.0,
        mismatch=mismatch,
        matching_amount=matching_amount,
        )
    
    return train_generator, val_generator, test_generator, initial_bias