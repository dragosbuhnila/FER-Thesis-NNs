import os
import numpy as np
from PIL import Image
import h5py
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.config import EMOTIONS
from modules.landmark_utils import detect_facial_landmarks, get_landmark_coordinate_sets_by_emotion__batch, load_landmark_coordinates
from modules.mask import apply_mask_to__batch
from modules.misc import hash_image



def generate_h5_from_images(test_set_path, resized_path, h5_path):
    class_names = sorted(os.listdir(test_set_path))
    paths = []
    X_test = []
    y_test = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(test_set_path, class_name)
        image_files = sorted(os.listdir(class_folder))
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            # Load PNG image as RGB NumPy array and resize to (128, 128, 3)
            image = np.array(Image.open(image_path).convert('RGB').resize((128, 128)))
            X_test.append(image)
            y_test.append(class_idx)  # Store class index instead of name
            paths.append(image_path)

            # Also save the images to resized_path
            save_folder = os.path.join(resized_path, class_name)
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(save_folder, image_file))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 3) Save new h5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)  # Now integers
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))  # Save as bytes
        f.create_dataset("paths", data=np.array(paths).astype('S'))  # Save as bytes
    print(f"Saved {X_test.shape[0]} images to {h5_path}")

    # 4) Check saved h5 to have 350 images, 7 classes, 50 images per class
    with h5py.File(h5_path, "r") as f:
        if "X_test" not in f.keys():
            raise ValueError("X_test not found in the H5 file.")
        if "y_test" not in f.keys():
            raise ValueError("y_test not found in the H5 file.")
        if "class_names" not in f.keys():
            raise ValueError("class_names not found in the H5 file.")
        if "paths" not in f.keys():
            raise ValueError("paths not found in the H5 file.")
        
        X_test_loaded = np.array(f["X_test"])
        y_test_loaded = np.array(f["y_test"])
        class_names_loaded = [name.decode('utf-8') for name in f["class_names"][...]]
        paths_loaded = [path.decode('utf-8') for path in f["paths"][...]]

        if X_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 images, but found {X_test_loaded.shape[0]}.")
        if y_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 labels, but found {y_test_loaded.shape[0]}.")
        if len(class_names_loaded) != 7:
            raise ValueError(f"Expected 7 classes, but found {len(class_names_loaded)}.")
        if len(paths_loaded) != 350:
            raise ValueError(f"Expected 350 paths, but found {len(paths_loaded)}.")

class RandomOcclusion(keras.layers.Layer):
    def __init__(self, occlusion_probability, masking_function, **kwargs):
        super().__init__(**kwargs)
        self.occlusion_probability = occlusion_probability
        self.masking_function = masking_function

    def call(self, images, labels, image_landmarks, training=None):
        # TODO: check if this really is needed
        if training is None:
            training = keras.backend.learning_phase()

        def occlude():
            # images is a tf.Tensor:
            # TODO make work as is instead of converting to numpy
            numpy_images = images.numpy()

            # a) Get all the landmarks, in the form of coordinates, for each image
            #       i.e. for each image every single face point, even unnecessary ones, will be there (they should be cached already)
            # ____________________________________________________________________________________________
            # If no landmarks are detected, then occlusion isn't possible for the approach we're currently using (i.e. masking based on AU landmarks),
            #   so leave the unprocessable image out and reinsert it with no occlusion
            error_indices = []
            for i, landmarks_all in enumerate(image_landmarks):
                if len(landmarks_all) == 0:
                    error_indices.append(i)

            unoccludable_images = dict()
            # sort error indices in reverse order so that the insertion has correct indices (deleting from the end first)
            error_indices.sort(reverse=True)
            for i in error_indices:
                # Save the image without occlusion and remove it from the batch
                unoccludable_images[i] = numpy_images[i]
                image_landmarks = np.delete(image_landmarks, i, axis=0) 
            # ____________________________________________________________________________________________           
            
            # b) Get the coordinates relating to just the specific emotion needed, for each image
            # ____________________________________________________________________________________________
            emotions = [EMOTIONS[label] for label in labels]
            list_of_landmark_sets = get_landmark_coordinate_sets_by_emotion__batch(image_landmarks, emotions)
            # ____________________________________________________________________________________________

            # c) Apply occlusion based on the landmarks
            # ____________________________________________________________________________________________
            occluded = apply_mask_to__batch(numpy_images, list_of_landmark_sets, self.masking_function)
            # ____________________________________________________________________________________________

            # a2) Reinsert unoccludable images without occlusion
            # sort unoccludable images by key in reverse order (so double reverse means original order) so that the insertion has correct indices (inserting from the start again)
            for i, img in sorted(unoccludable_images.items()):
                occluded = np.insert(occluded, i, img, axis=0)

            # If I implemented the pipeline correctly, when cache miss doesn't happen this is already a tensor, else it's a list
            # But since the augmentation pipeline is not GPU compatible RN, I convert to numpy before passing to augmentation
            if tf.is_tensor(occluded):
                return occluded.numpy()
            elif isinstance(occluded, list):
                occluded = np.array(occluded)
                return occluded
            else:
                raise TypeError("Occluded images must be either a tf.Tensor or a list, there must have been an issue.")

        return tf.cond(
            tf.less(tf.random.uniform([]), self.occlusion_probability),
            occlude,
            lambda: images
        )

class CustomBalancedDataGenerator(Sequence):
    def __init__(self, x_data, y_data, x_hashes, x_landmarks, batch_size, occlusion_probability, masking_function, augmentations=None, data_inf=None, label_smoothing=0.1, paths_data=None, **kwargs):
        super().__init__(**kwargs)
        if data_inf not in ['train', 'valid', 'test']:
            raise ValueError(f"data_inf must be 'train', 'valid', or 'test', but is '{data_inf}'")

        self.x_data = x_data
        self.y_data = y_data
        self.x_hashes = x_hashes
        self.x_landmarks = x_landmarks
        self.paths_data = paths_data
        self.indices = np.arange(len(x_data))
        
        self.data_inf = data_inf
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.occlusion_layer = RandomOcclusion(occlusion_probability, masking_function)

        if data_inf in ['train', 'valid']:
            #print(y_data)
            # 1) Augment
            self.augmentations = ImageDataGenerator(**augmentations)

            # 2) Class balancing
            self.classes = np.unique(np.argmax(y_data, axis=1))  # Ricaviamo le classi dai dati one-hot encoded
            self.class_indices = {cls: np.where(np.argmax(y_data, axis=1) == cls)[0] for cls in self.classes}
            self.num_classes = len(self.classes)
            self.samples_per_class = max(1, self.batch_size // self.num_classes)
            self.class_pointers = {cls: 0 for cls in self.classes} # Coda ciclica per le classi minoritarie
        elif data_inf == 'test':
            self.augmentations = ImageDataGenerator(**(augmentations or {}))
        
        self.index = 0
        self.on_epoch_end() # Shuffles data by shuffling indices (only in train/valid)
        print(f"Generator initialized: {data_inf} mode")

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __next__(self):
        # Il comportamento dell'iteratore
        if self.index >= len(self):
            raise StopIteration
        batch = self.__getitem__(self.index)
        self.index += 1
        return batch

    def __iter__(self):
        # Rende l'oggetto un iteratore
        self.index = 0
        return self
    
    def __getitem__(self, index):
        if self.data_inf == 'test':
            # Per il test set, usiamo semplicemente gli indici
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.x_data))

            batch_x = self.x_data[start_idx:end_idx]
            batch_y = self.y_data[start_idx:end_idx]
            batch_x_hashes = self.x_hashes[start_idx:end_idx]
            batch_x_landmarks = self.x_landmarks[start_idx:end_idx]
            batch_paths = self.paths_data[start_idx:end_idx] if self.paths_data is not None else [None]*len(batch_x)
        elif self.data_inf in ['train', 'valid']:
            # Per train/valid, selezioniamo batch bilanciati
            batch_x, batch_y, batch_x_hashes, batch_x_landmarks, batch_paths = [], [], [], [], []
            for cls in self.classes:
                cls_indices = self.class_indices[cls]
                cls_pointer = self.class_pointers[cls]

                # Select indices from the class circular queue
                selected_indices = cls_indices[cls_pointer:cls_pointer + self.samples_per_class]
                selected_indices = np.asarray(selected_indices, dtype=int)

                # Debug info (only prints if unusual types are present or sel is empty)
                if selected_indices.size == 0:
                    # nothing to take for this class this round
                    # keep class pointer unchanged (pointer update below uses len(sel) so safe)
                    continue

                # Helper to index arrays or lists safely
                def safe_index(container, indices):
                    if isinstance(container, np.ndarray):
                        return container[indices]
                    else:
                        # container is likely a list: convert indices to Python ints and use list comprehension
                        return [container[int(i)] for i in indices]

                try:
                    batch_x.extend(safe_index(self.x_data, selected_indices))
                    batch_y.extend(safe_index(self.y_data, selected_indices))
                    batch_x_hashes.extend(safe_index(self.x_hashes, selected_indices))
                    batch_x_landmarks.extend(safe_index(self.x_landmarks, selected_indices))
                    if self.paths_data is not None:
                        batch_paths.extend(safe_index(self.paths_data, selected_indices))
                    else:
                        batch_paths.extend([None] * len(selected_indices))
                except Exception as exc:
                    # Print full diagnostics to help debug the specific failing case
                    print("DEBUG: failure indexing during batch construction")
                    print(f"  cls={cls!r}, cls_pointer={cls_pointer}, samples_per_class={self.samples_per_class}")
                    print(f"  cls_indices type={type(cls_indices)}, len={len(cls_indices)}")
                    print(f"  sel (np.asarray) dtype={selected_indices.dtype}, shape={selected_indices.shape}, values={selected_indices.tolist()[:20]}")
                    print(f"  x_data type={type(self.x_data)}, len={len(self.x_data) if hasattr(self.x_data,'__len__') else 'n/a'}")
                    print(f"  x_landmarks type={type(self.x_landmarks)}, len={len(self.x_landmarks) if hasattr(self.x_landmarks,'__len__') else 'n/a'}")
                    raise

                # Aggiorna il puntatore per la classe
                self.class_pointers[cls] += len(selected_indices)

                # Se abbiamo esaurito i dati per la classe, fai uno shuffle e riparti
                if self.class_pointers[cls] >= len(cls_indices):
                    self.class_pointers[cls] = 0
                    np.random.shuffle(cls_indices)  # Shuffle della classe
                    self.class_indices[cls] = cls_indices

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            batch_x_hashes = np.array(batch_x_hashes)
            batch_x_landmarks = np.array(batch_x_landmarks)
            batch_paths = np.array(batch_paths)

            batch_x, batch_y, batch_x_hashes, batch_x_landmarks, batch_paths = shuffle(batch_x, batch_y, batch_x_hashes, batch_x_landmarks, batch_paths)

            # Applica il label smoothing
            if self.label_smoothing > 0:
                batch_y = self.apply_label_smoothing(batch_y)

        # Applica il rescale o le trasformazioni per augmentation
        batch_x = self.occlusion_layer(batch_x, np.argmax(batch_y, axis=1), batch_x_landmarks, training=(self.data_inf != 'test'))
        augmented_batch_x = np.zeros_like(batch_x)
        for i in range(len(batch_x)):
            augmented_batch_x[i] = self.augmentations.random_transform(batch_x[i])

        return augmented_batch_x, batch_y, batch_paths, batch_x_hashes

    def on_epoch_end(self):
        if self.data_inf != 'test':
            print("Epoch ended. Shuffling data.")
            for cls in self.classes:
                np.random.shuffle(self.class_indices[cls])  # Shuffle degli indici per ogni classe

    def apply_label_smoothing(self, labels):
        """Applica il label smoothing alle etichette one-hot"""
        if self.label_smoothing > 0:
            labels = labels.astype(np.float32)  # Assicurati che sia in formato float
            num_classes = labels.shape[1]  # Ottieni il numero di classi (assumendo one-hot encoding)
            smooth_value = self.label_smoothing / (num_classes - 1)  # Calcolo del valore per le classi non corrette
            smoothed_labels = np.ones_like(labels, dtype=np.float32) * smooth_value  # Etichette smussate per tutte le classi
            for i in range(len(labels)):
                true_class = np.argmax(labels[i])  # Ottieni la classe corretta (indice della classe 1)
                smoothed_labels[i, true_class] = 1.0 - self.label_smoothing  # Imposta la probabilità della classe corretta
            return smoothed_labels
        else:
            return labels

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

def load_test_generator(path, occlusion_probability, masking_function):
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
        batch_size=64,        
        augmentations={},
        label_smoothing=0,

        masking_function=masking_function,
        occlusion_probability=occlusion_probability,
    )

    return data_generator

def load_data_generators(train_path, test_path, occlusion_probability, masking_function, use_label_smoothing, debug=False):
    # 1) Load training and validation data
    # ____________________________________
    X_train, y_train, X_val, y_val, trainval_class_names, train_paths_data, val_paths_data = load_data_and_labels(train_path, 'train')
    X_test, y_test, test_class_names, test_paths_data = load_data_and_labels(test_path, 'test')

    if debug:
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
    if debug:
        X_train_landmarks = np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_train, X_train_hashes)])
        X_val_landmarks =   np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_val, X_val_hashes)])
        X_test_landmarks =  np.array([detect_facial_landmarks(img, img_hash, False, True, True) for img, img_hash in zip(X_test, X_test_hashes)])
    else:
        X_train_landmarks = np.array([load_landmark_coordinates(X_train_hash) for X_train_hash in X_train_hashes])
        X_val_landmarks =   np.array([load_landmark_coordinates(X_val_hash) for X_val_hash in X_val_hashes])
        X_test_landmarks =  np.array([load_landmark_coordinates(X_test_hash) for X_test_hash in X_test_hashes])
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
    print(f"After filtering zero-length landmarks: X_train length: {len(X_train)}, X_val length: {len(X_val)}, X_test length: {len(X_test)}")

    # 3) Compute initial bias
    # ____________________________________
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_probabilities = class_counts / total_samples
    initial_bias = np.log(class_probabilities / (1 - class_probabilities))
    # print("Bias iniziale per ciascuna classe:", initial_bias)

    # 3) Shuffle training and validation data
    # ____________________________________
    if train_paths_data is not None:
        X_train, y_train, X_train_hashes, train_paths_data = shuffle(X_train, y_train, X_train_hashes, train_paths_data)
        X_val, y_val, X_val_hashes, val_paths_data = shuffle(X_val, y_val, X_val_hashes, val_paths_data)
    else:
        X_train, y_train, X_train_hashes = shuffle(X_train, y_train, X_train_hashes)
        X_val, y_val, X_val_hashes = shuffle(X_val, y_val, X_val_hashes)
    
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
        batch_size=64,
        augmentations=train_augmentations,
        label_smoothing=label_smoothing_value,
        masking_function=masking_function,
        occlusion_probability=occlusion_probability,
        )
    val_generator = CustomBalancedDataGenerator(
        x_data=X_val,
        y_data=y_val_one_hot,
        x_hashes=X_val_hashes,
        x_landmarks=X_val_landmarks,
        paths_data=val_paths_data,
        data_inf='valid',
        batch_size=64,
        augmentations=train_augmentations,
        label_smoothing=0,
        masking_function=masking_function,
        occlusion_probability=1.0,
        )
    test_generator = CustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        x_hashes=X_test_hashes,
        x_landmarks=X_test_landmarks,
        paths_data=test_paths_data,
        data_inf='test',
        batch_size=64,
        augmentations={},
        label_smoothing=0,
        masking_function=masking_function,
        occlusion_probability=1.0,
        )
    
    return train_generator, val_generator, test_generator, initial_bias