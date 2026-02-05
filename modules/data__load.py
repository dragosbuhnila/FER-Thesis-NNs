import os
from joblib import Parallel, delayed
import numpy as np
import h5py
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import time

from modules.config import BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT, EMOTIONS, LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE, LANDMARK_COORDINATES_FOLDER_PATH
from modules.data import OfflineOcclusionGenerator, OnlineOcclusionGenerator, OldCustomBalancedDataGenerator
from modules.data__load__misc import load_data_and_labels, remove_indices_from_data
from modules.landmark_utils import detect_facial_landmarks, load_landmark_coordinates
from modules.misc import hash_image



# ==================================================================================================
# ===============================   Online loading functions =======================================
# ======================    load test, load train, load val, load all ==============================
# ==================================================================================================

def load_online_test_generator(path, batch_size=64):
    """
    Load only the test generator from a test H5 file (path).
    """
    X_test, y_test, class_names, test_paths = load_data_and_labels(path, 'test')

    # Test has no dupes, so no need to remove them
    # if remove_dupes:
    #     for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
    #         if split == 'X_test':
    #             X_test, y_test, test_paths = remove_indices_from_data(X_test, y_test, test_paths, indices)

    for emotion in EMOTIONS:
        if emotion not in class_names:
            raise ValueError(f"Class '{emotion}' not found in class names from H5 file.")
    NUM_CLASSES = len(class_names)

    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)
    # X_test_hashes = np.array([hash_image(img) for img in X_test])

    # if parallelize:
    #     X_test_landmarks = np.array(Parallel(n_jobs=-1)(
    #         delayed(load_landmark_coordinates)(h) for h in tqdm(X_test_hashes, desc="Loading test landmarks")
    #     ))
    # else:
    #     X_test_landmarks = np.array([load_landmark_coordinates(h) for h in tqdm(X_test_hashes, desc="Loading test landmarks")])

    # valid_indices = [i for i, lm in enumerate(X_test_landmarks) if len(lm) > 0]
    # if len(valid_indices) != len(X_test):
    #     X_test = X_test[valid_indices]
    #     y_test_one_hot = y_test_one_hot[valid_indices]
    #     # X_test_hashes = X_test_hashes[valid_indices]
    #     # X_test_landmarks = X_test_landmarks[valid_indices]
    #     test_paths = test_paths[valid_indices] if test_paths is not None else None

    data_generator = OldCustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        data_inf='test',
        batch_size=batch_size,
        augmentations={},
        label_smoothing=0,
    )

    return data_generator


def load_online_train_generator(train_path, occlusion_probability, masking_function_name, mismatch, batch_size=64, parallelize=False, remove_dupes=True, matching_amount=0.2, label_smoothing=0.0):
    """
    Load only the train generator from a train H5 file (train_path).
    """
    X_train, y_train, X_val, y_val, trainval_class_names, train_paths_data, val_paths_data = load_data_and_labels(train_path, 'train')

    if remove_dupes:
        for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
            if split == 'X_train':
                X_train, y_train, train_paths_data = remove_indices_from_data(X_train, y_train, train_paths_data, indices)

    for emotion in EMOTIONS:
        if emotion not in trainval_class_names:
            raise ValueError(f"Class '{emotion}' not found in training/validation class names from H5 file.")

    NUM_CLASSES = len(trainval_class_names)
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    X_train_hashes = np.array([hash_image(img) for img in X_train])

    if parallelize:
        X_train_landmarks = np.array(Parallel(n_jobs=-1)(
            delayed(load_landmark_coordinates)(h) for h in tqdm(X_train_hashes, desc="Loading training landmarks")
        ))
    else:
        X_train_landmarks = np.array([load_landmark_coordinates(h) for h in tqdm(X_train_hashes, desc="Loading training landmarks")])

    valid_indices = [i for i, lm in enumerate(X_train_landmarks) if len(lm) > 0]
    if len(valid_indices) != len(X_train):
        X_train = X_train[valid_indices]
        y_train_one_hot = y_train_one_hot[valid_indices]
        X_train_hashes = X_train_hashes[valid_indices]
        X_train_landmarks = X_train_landmarks[valid_indices]
        train_paths_data = train_paths_data[valid_indices] if train_paths_data is not None else None

    train_augmentations = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'shear_range': 0.3,
        'horizontal_flip': True,
        'fill_mode': 'wrap',
    }

    train_generator = OnlineOcclusionGenerator(
        x_data=X_train,
        y_data=y_train_one_hot,
        x_hashes=X_train_hashes,
        x_landmarks=X_train_landmarks,
        paths_data=train_paths_data,
        data_inf='train',
        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=label_smoothing,
        masking_function_name=masking_function_name,
        occlusion_probability=occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
    )

    return train_generator


def load_online_valid_generator(train_path, occlusion_probability, masking_function_name, mismatch, batch_size=64, parallelize=False, remove_dupes=True, matching_amount=0.2):
    """
    Load only the validation generator from a train H5 file (train_path).
    """
    X_train, y_train, X_val, y_val, trainval_class_names, train_paths_data, val_paths_data = load_data_and_labels(train_path, 'train')

    if remove_dupes:
        for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
            if split == 'X_val':
                X_val, y_val, val_paths_data = remove_indices_from_data(X_val, y_val, val_paths_data, indices)

    for emotion in EMOTIONS:
        if emotion not in trainval_class_names:
            raise ValueError(f"Class '{emotion}' not found in training/validation class names from H5 file.")

    NUM_CLASSES = len(trainval_class_names)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    X_val_hashes = np.array([hash_image(img) for img in X_val])

    if parallelize:
        X_val_landmarks = np.array(Parallel(n_jobs=-1)(
            delayed(load_landmark_coordinates)(h) for h in tqdm(X_val_hashes, desc="Loading validation landmarks")
        ))
    else:
        X_val_landmarks = np.array([load_landmark_coordinates(h) for h in tqdm(X_val_hashes, desc="Loading validation landmarks")])

    # filter out entries with zero-length landmarks
    valid_indices = [i for i, lm in enumerate(X_val_landmarks) if len(lm) > 0]
    if len(valid_indices) != len(X_val):
        X_val = X_val[valid_indices]
        y_val_one_hot = y_val_one_hot[valid_indices]
        X_val_hashes = X_val_hashes[valid_indices]
        X_val_landmarks = X_val_landmarks[valid_indices]
        val_paths_data = val_paths_data[valid_indices] if val_paths_data is not None else None

    augmentations = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'shear_range': 0.3,
        'horizontal_flip': True,
        'fill_mode': 'wrap',
    }

    val_generator = OnlineOcclusionGenerator(
        x_data=X_val,
        y_data=y_val_one_hot,
        x_hashes=X_val_hashes,
        x_landmarks=X_val_landmarks,
        paths_data=val_paths_data,
        data_inf='valid',
        batch_size=batch_size,
        augmentations=augmentations,
        label_smoothing=0,
        masking_function_name=masking_function_name,
        occlusion_probability=occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
    )

    return val_generator


def load_online_data_generators(trainval_path, test_path, training_occlusion_probability, masking_function_name, use_label_smoothing, mismatch, small_subset=False, run_detection=False, remove_dupes=True, parallelize=True, matching_amount=0.2, batch_size=64, validation_occlusion_probability=0.5, pos_or_neg=None, dont_augment=False, dont_rebalance_trainval=False):
    # 1) Load training and validation data
    # ____________________________________
    X_train, y_train, X_val, y_val, trainval_class_names, train_paths_data, val_paths_data = load_data_and_labels(trainval_path, 'train')
    X_test, y_test, test_class_names, test_paths_data = load_data_and_labels(test_path, 'test')

    if remove_dupes:
        for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
            if split == 'X_train':
                X_train, y_train, train_paths_data = remove_indices_from_data(X_train, y_train, train_paths_data, indices)
            elif split == 'X_val':
                X_val, y_val, val_paths_data = remove_indices_from_data(X_val, y_val, val_paths_data, indices)
            # elif split == 'X_test':
            #     X_test, y_test, test_paths_data = remove_indices_from_data(X_test, y_test, test_paths_data, indices)

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
    # X_test_hashes = np.array([hash_image(img) for img in X_test])
    
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
        raise ValueError(f"Training paths are provided but validation paths are missing, which is weird, so check the dataset at {trainval_path}.")
    if train_paths_data is None and val_paths_data is not None:
        raise ValueError(f"Validation paths are provided but training paths are missing, which is weird, so check the dataset at {trainval_path}.")
    
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
            # X_test_landmarks = np.array(Parallel(n_jobs=-1)(    delayed(detect_facial_landmarks)(img, img_hash, False, True, True) for img, img_hash in tqdm(zip(X_test, X_test_hashes), desc="Detecting test landmarks")          ))
        else:
            X_train_landmarks = np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_train, X_train_hashes)])
            X_val_landmarks =   np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_val, X_val_hashes)])
            # X_test_landmarks =  np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_test, X_test_hashes)])
    else:
        if parallelize:
            X_train_landmarks = np.array(Parallel(n_jobs=-1)(   delayed(load_landmark_coordinates)(X_train_hash)    for X_train_hash in tqdm(X_train_hashes, desc="Loading training landmarks")     ))
            X_val_landmarks = np.array(Parallel(n_jobs=-1)(     delayed(load_landmark_coordinates)(X_val_hash)      for X_val_hash in   tqdm(X_val_hashes, desc="Loading validation landmarks")     ))
            # X_test_landmarks = np.array(Parallel(n_jobs=-1)(    delayed(load_landmark_coordinates)(X_test_hash)     for X_test_hash in  tqdm(X_test_hashes, desc="Loading test landmarks")          ))
        else:
            X_train_landmarks = np.array([load_landmark_coordinates(X_train_hash) for X_train_hash in tqdm(X_train_hashes, desc="Loading training landmarks")])
            X_val_landmarks =   np.array([load_landmark_coordinates(X_val_hash)   for X_val_hash in   tqdm(X_val_hashes, desc="Loading validation landmarks")])
            # X_test_landmarks =  np.array([load_landmark_coordinates(X_test_hash)  for X_test_hash in  tqdm(X_test_hashes, desc="Loading test landmarks")])
    print(f"Landmarks detected for training and validation sets. X_train_landmarks length: {len(X_train_landmarks)}, X_val_landmarks length: {len(X_val_landmarks)}")

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
    # X_test, y_test, X_test_hashes, X_test_landmarks, test_paths_data = filter_zero_length_landmarks(X_test, y_test, X_test_hashes, X_test_landmarks, test_paths_data, name="test")
    
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
    if dont_augment:
        train_augmentations = {}
    else:
        train_augmentations = {
            'rotation_range': 10,
            'width_shift_range': 0.2,
            'shear_range': 0.3,
            'horizontal_flip': True,
            # 'fill_mode': 'wrap', # Idk why this was here but it makes zero sense, 
            # 'fill_mode': 'nearest', # Distorst the image if it's near the edges
            'fill_mode': 'constant', # Fill with zeros (black)
        }

    # 6) Label smoothing
    # ____________________________________
    if use_label_smoothing:
        label_smoothing_value = 0.05
    else:
        label_smoothing_value = 0.0

    # 7) Create generators
    # ____________________________________
    train_generator = OnlineOcclusionGenerator(
        x_data=X_train, 
        y_data=y_train_one_hot,
        x_hashes=X_train_hashes,
        x_landmarks=X_train_landmarks,
        paths_data=train_paths_data,
        data_inf='train',
        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=label_smoothing_value,
        masking_function_name=masking_function_name,
        occlusion_probability=training_occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
        pos_or_neg=pos_or_neg,
        dont_rebalance_trainval=dont_rebalance_trainval,
        )
    val_generator = OnlineOcclusionGenerator(
        x_data=X_val,
        y_data=y_val_one_hot,
        x_hashes=X_val_hashes,
        x_landmarks=X_val_landmarks,
        paths_data=val_paths_data,
        data_inf='valid',
        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=0,
        masking_function_name=masking_function_name,
        occlusion_probability=validation_occlusion_probability,
        mismatch=mismatch,
        matching_amount=matching_amount,
        pos_or_neg=pos_or_neg,
        dont_rebalance_trainval=dont_rebalance_trainval,
        )
    test_generator = OldCustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        data_inf='test',
        batch_size=batch_size,
        augmentations={},
        label_smoothing=0,
    )
    
    return train_generator, val_generator, test_generator, initial_bias

# ==================================================================================================
# ==============================   Offline loading functions =======================================
# ======================    load test, load train, load val, load all ==============================
# ==================================================================================================

def generate_occlusion_indexer(X_occ: np.ndarray, X_train_occ_original_hashes: np.ndarray, occs: np.ndarray, pos_or_negs: np.ndarray):
    # Add parallelization for both cpu and tf gpu later
    occlusion_indexer = {}
    types_of_occlusion = set()
    for img, original_hash, occ, pos_or_neg in zip(X_occ, X_train_occ_original_hashes, occs, pos_or_negs):
        # Type hints for variables
        # img: np.ndarray
        # original_hash: str
        # occ: int
        # pos_or_neg: int
        if original_hash not in occlusion_indexer:
            occlusion_indexer[original_hash] = {}

        occlusion_type = f"{occ}_{pos_or_neg}"
        types_of_occlusion.add(occlusion_type)
        occlusion_indexer[original_hash][occlusion_type] = img

    return occlusion_indexer, types_of_occlusion

def load_offline_data_generators(original_trainval_path: str, occluded_trainval_path: str, occluded_test_path: str,
                                training_occlusion_probability: float = 0.8, validation_occlusion_probability: float = 1.0, # matching_amount=0.2, pos_or_neg=None
                                masking_function_name: str = "lines", use_label_smoothing: bool = True, 
                                small_subset=False, batch_size=64, dont_augment=False, dont_rebalance_trainval=False):
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Loading data from H5 files...", flush=True)
    # 1) Load training and validation data
    # ____________________________________________________________________________________________________________________________________________
    X_train, y_train, X_val, y_val, trainval_class_names    = load_data_and_labels(original_trainval_path, 'train')
    X_test, y_test, test_class_names                        = load_data_and_labels(occluded_test_path, 'test')
    X_train_occ, y_train_occ, X_val_occ, y_val_occ, _, X_train_occ_original_hashes, X_val_occ_original_hashes, occ_train, mismatch_train, pos_or_neg_train, occ_val, mismatch_val, pos_or_neg_val = load_data_and_labels(occluded_trainval_path, 'train', occlusion_dataset=True)
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Loaded data from: original train and validation dataset, occluded train and validation dataset, occluded test dataset.", flush=True)

    for emotion in EMOTIONS:
        if emotion not in trainval_class_names:
            raise ValueError(f"Class '{emotion}' not found in training/validation class names from H5 file.")
        if emotion not in test_class_names:
            raise ValueError(f"Class '{emotion}' not found in test class names from H5 file.")
    class_names = test_class_names


    # 2) Remove duplicates and unlandmarkable images
    # ____________________________________________________________________________________________________________________________________________
    for split, indices in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT.items():
        if split == 'X_train':
            X_train, y_train, _ =      remove_indices_from_data(X_train, y_train, None, indices)
        elif split == 'X_val':
            X_val, y_val, _ =          remove_indices_from_data(X_val, y_val, None, indices)
        # elif split == 'X_test':
        #     X_test, y_test, _ =      remove_indices_from_data(X_test, y_test, None, indices)

    # 3) Small subset for debugging
    # ____________________________________________________________________________________________________________________________________________
    if small_subset:
        debug_limit = 100
        X_train =           X_train[:debug_limit]
        y_train =           y_train[:debug_limit]
        X_val =             X_val[:debug_limit]
        y_val =             y_val[:debug_limit]
        X_test =            X_test[:debug_limit]
        y_test =            y_test[:debug_limit]


    # 4) Make occlusion indexer
    # ____________________________________________________________________________________________________________________________________________
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Generating hashes occlusion indexers for training and validation sets...", flush=True)
    X_train_hashes =    np.array([hash_image(img) for img in X_train])
    X_val_hashes =      np.array([hash_image(img) for img in X_val])
    # X_test_hashes =   np.array([hash_image(img) for img in X_test])

    train_occlusion_indexer, types_of_occlusion = generate_occlusion_indexer(X_train_occ, X_train_occ_original_hashes, occ_train, pos_or_neg_train)
    val_occlusion_indexer, _ =   generate_occlusion_indexer(X_val_occ, X_val_occ_original_hashes, occ_val, pos_or_neg_val)
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Generated occlusion indexers for training and validation sets.", flush=True)

    # Validate that all expected occlusion types are present
    position_of_neutral = EMOTIONS.index("NEUTRAL")
    emotion_indices = list(range(len(EMOTIONS)))
    emotions_indices_without_neutral = emotion_indices[:position_of_neutral] + emotion_indices[position_of_neutral+1:]
    expected_types_of_occlusions = [f"{occ}_1" for occ in emotions_indices_without_neutral] + [f"{occ}_0" for occ in emotions_indices_without_neutral]
    for expected_type in expected_types_of_occlusions:
        if expected_type not in types_of_occlusion:
            error_message = f"Expected occlusion type '{expected_type}' not found in occlusion indexer."
            error_message += f"\nExpected types of occlusion: {expected_types_of_occlusions}"
            error_message += f"\nActual types of occlusion found: {types_of_occlusion}"
            error_message += f"\nOcclusion indexer keys"
            for key in train_occlusion_indexer.keys():
                error_message += f"\n  {key}: {list(train_occlusion_indexer[key].keys())}"
            raise ValueError(error_message)


    # 5) Compute initial bias
    # ____________________________________________________________________________________________________________________________________________
    try:
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_probabilities = class_counts / total_samples
        initial_bias = np.log(class_probabilities / (1 - class_probabilities))
    except Exception as e:
        print(f"[ERROR] {time.strftime('%Y%m%d-%H%M%S')} Error computing initial bias: {e}. class_counts: {class_counts}, total_samples: {total_samples}.", flush=True)
        raise
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Initial bias computed. class_counts: {class_counts}, total_samples: {total_samples}. class_probabilities: {class_probabilities}, initial_bias: {initial_bias}", flush=True)


    # 6) Shuffle training and validation data
    # _____________________________________________________________________________________________________________________________________________
    X_train, y_train, X_train_hashes = shuffle(X_train, y_train, X_train_hashes)
    X_val, y_val, X_val_hashes = shuffle(X_val, y_val, X_val_hashes)


    # 7) One-hot encoding, augmentations, generators
    # ____________________________________________________________________________________________________________________________________________
    NUM_CLASSES = len(class_names)
    y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_one_hot = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)


    # 8) Augmentations
    # ____________________________________________________________________________________________________________________________________________
    if dont_augment:
        train_augmentations = {}
    else:
        train_augmentations = {
            'rotation_range': 10,
            'width_shift_range': 0.2,
            'shear_range': 0.3,
            'horizontal_flip': True,
            # 'fill_mode': 'wrap', # Idk why this was here but it makes zero sense, 
            # 'fill_mode': 'nearest', # Distorst the image if it's near the edges
            'fill_mode': 'constant', # Fill with zeros (black)
        }


    # 9) Generators
    # ____________________________________________________________________________________________________________________________________________
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} All set. Creating data generators...", flush=True)
    train_generator = OfflineOcclusionGenerator(
        split='train',
        x_data=None, 
        y_data=None,
        x_hashes=None,
        occlusion_indexer=train_occlusion_indexer,
        types_of_occlusion=types_of_occlusion,

        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=0.05 if use_label_smoothing else 0.0,

        masking_function_name=masking_function_name,
        occlusion_probability=training_occlusion_probability,
        dont_rebalance_trainval=dont_rebalance_trainval,
        )
    val_generator = OfflineOcclusionGenerator(
        split='valid',
        x_data=None,
        y_data=None,
        x_hashes=None,
        occlusion_indexer=val_occlusion_indexer,
        types_of_occlusion=types_of_occlusion,

        batch_size=batch_size,
        augmentations=train_augmentations,
        label_smoothing=0,

        masking_function_name=masking_function_name,
        occlusion_probability=validation_occlusion_probability,
        dont_rebalance_trainval=dont_rebalance_trainval,
        )
    test_generator = OldCustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        data_inf='test',
        batch_size=batch_size,
        augmentations={},
        label_smoothing=0,
    )
    print(f"[INFO] {time.strftime('%Y%m%d-%H%M%S')} Data generators created.", flush=True)
    
    return train_generator, val_generator, test_generator, initial_bias