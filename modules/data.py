import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.config import EMOTIONS
from modules.landmark_utils import get_landmark_coordinate_sets_by_emotion__batch, load_landmark_coordinates
from modules.mask import apply_mask_to__batch
from modules.visualize import plot_image



DEBUG_OCCLUSION = False



class RandomOcclusion(keras.layers.Layer):
    def __init__(self, occlusion_probability, masking_function, mismatch, matching_amount, **kwargs):
        super().__init__(**kwargs)
        self.occlusion_probability = occlusion_probability
        self.masking_function = masking_function
        self.mismatch = mismatch
        self.matching_amount = matching_amount

    def call(self, images, labels, image_landmarks, image_hashes, training=None):
        # labels = tf.convert_to_tensor(labels) if not tf.is_tensor(labels) else labels
        # image_landmarks = tf.convert_to_tensor(image_landmarks) if not tf.is_tensor(image_landmarks) else image_landmarks

        # TODO: check if this really is needed AND if training not being a tf.Tensor slows things down
        if training is None:
            training = keras.backend.learning_phase()
        
        images = images.numpy()

        def occlude(images, labels, image_landmarks, image_hashes, positive_or_negative_batch):
            # images is a np array

            # a) Get all the landmarks, in the form of coordinates, for each image
            #       i.e. for each image every single face point, even unnecessary ones, will be there (they should be cached already)
            # ____________________________________________________________________________________________
            # If no landmarks are detected, then occlusion isn't possible for the approach we're currently using (i.e. masking based on AU landmarks),
            #   so leave the unprocessable image out and reinsert it with no occlusion
            error_indices = []
            for i, landmarks_all in enumerate(image_landmarks):
                if len(landmarks_all) == 0:
                    error_indices.append(i)

            # # a1) Prepare unoccludable images to be reinserted later (won't do it in this version as I am already removing the problem images)
            # unoccludable_images = dict()
            # # sort error indices in reverse order so that the insertion has correct indices (deleting from the end first)
            # error_indices.sort(reverse=True)
            # for i in error_indices:
            #     # Save the image without occlusion and remove it from the batch
            #     unoccludable_images[i] = numpy_images[i]
            #     image_landmarks = np.delete(image_landmarks, i, axis=0) 
            # # ____________________________________________________________________________________________           
            
            # b) Get the coordinates relating to just the specific emotion needed, for each image
            # ____________________________________________________________________________________________
            emotions = [EMOTIONS[label] for label in labels]
            list_of_landmark_sets = get_landmark_coordinate_sets_by_emotion__batch(image_landmarks, emotions)
            # ____________________________________________________________________________________________

            # c) Apply occlusion based on the landmarks
            # ____________________________________________________________________________________________
            occluded = apply_mask_to__batch(images, list_of_landmark_sets, self.masking_function, positive_or_negative_batch)
            # ___________________________________________________________________________________________   

            # if DEBUG_OCCLUSION: # Visualize occluded images
            #     nop_variable = 0
            #     for image, landmarks_emotion, landmarks_emotions_index, hash in zip(occluded, emotions, labels, image_hashes):
            #         print(f"Hash for the current image: {hash}")
            #         landmarks_that_should_be = load_landmark_coordinates(hash)
            #         plot_image(image, title=f"landmarks for emotion {landmarks_emotion} (index {landmarks_emotions_index})\nHash: {hash}")
            #         nop_variable += 1

            # # a2) Reinsert unoccludable images without occlusion (won't do it in this version as I am already removing the problem images)
            # # sort unoccludable images by key in reverse order (so double reverse means original order) so that the insertion has correct indices (inserting from the start again)
            # for i, img in sorted(unoccludable_images.items()):
            #     occluded = np.insert(occluded, i, img, axis=0)

            # If I implemented the pipeline correctly, when cache miss doesn't happen this is already a tensor, else it's a list
            # But since the augmentation pipeline is not GPU compatible RN, I convert to numpy before passing to augmentation
            if tf.is_tensor(occluded):
                return occluded.numpy()
            elif isinstance(occluded, list):
                occluded = np.array(occluded)
                return occluded
            else:
                raise TypeError("Occluded images must be either a tf.Tensor or a list, there must have been an issue.")

        if self.mismatch:
            # Allow for mismatching emotion-landmarks (e.g. putting 'happy' landmarks on a 'sad' image)
            #   > overwrite labels to have a uniform distribution of emotions, except for NEUTRAL (4)
            num_emotions = len(EMOTIONS)  # Total number of emotions
            if num_emotions != 7:
                raise ValueError(f"Expected 7 emotions, but got {num_emotions}. Check EMOTIONS mapping.")

            # 1) Exclude NEUTRAL (4) from pool
            non_neutral_emotions = [i for i in range(num_emotions) if i != 4] 

            # Here there's the choice of keeping always ratios that mirror the test sets (i.e. always having a 20% of the masked images being matched and so on) or allow for the 
            #   non occluded images to substitute any of these, meaning that the ratio may fluctuate inside of the single batch, but should stay kind of consisten throughout the epochs instead.
            # I'll keep this choice of picking the occlusions first (mirroring ratio) and then selecting which images will be occluded or not, since it's the easier approach, 
            #   and should also grant variety to the descent towards optimality.
            # The same philosophy applies to the choice of positive/negative. The distribution within a single batch of every of the 10 kinds of occlusion (nof_occlusion_types = 5 * 2  # 5 occlusion types (each unmatched emotion), each with +/- occlusion)
            #   won't be exaclty uniform since negative/positive isn't really chained to the choice of the mismatching emotion in the following code, but again, 
            #   it should uniform itself throughout the epochs, plus add variability between batches.

            # keep **matching_amount** of labels the same, change the rest
            labels = labels.astype(int)
            num_to_match = int(len(labels) * self.matching_amount)
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            indices_to_change = indices[num_to_match:]
            # indices_to_change.sort()  # Sort when debugging to check correctness easily

            mismatching_labels = np.random.choice(non_neutral_emotions, size=len(indices_to_change))
            positive_or_negative_batch = np.ones_like(labels)
            positive_or_negative_batch[indices_to_change] = np.random.randint(0, 2, size=len(indices_to_change))
            
            labels[indices_to_change] = mismatching_labels


        batch_size = images.shape[0]
        num_to_occlude = int(batch_size * self.occlusion_probability)

        # Nothing to occlude
        if num_to_occlude == 0:
            return images

        # Randomly select indices to occlude
        all_indices = np.arange(batch_size)
        np.random.shuffle(all_indices)
        occlude_indices = all_indices[:num_to_occlude]

        # Slice sub-batch
        images_sub = images[occlude_indices]
        labels_sub = labels[occlude_indices]
        landmarks_sub = image_landmarks[occlude_indices]
        hashes_sub = image_hashes[occlude_indices]
        posneg_sub = positive_or_negative_batch[occlude_indices]

        # Apply occlusion only to sub-batch
        occluded_sub = occlude(images_sub, labels_sub, landmarks_sub, hashes_sub, posneg_sub)
        images[occlude_indices] = occluded_sub

        for image in images:
            if DEBUG_OCCLUSION: # Visualize occluded images
                nop_variable = 0
                for image, landmarks_emotions_index, hash in zip(images, labels, image_hashes):
                    print(f"Hash for the current image: {hash}")
                    landmarks_that_should_be = load_landmark_coordinates(hash)
                    plot_image(image, title=f"landmarks for emotion {EMOTIONS[landmarks_emotions_index]} (index {landmarks_emotions_index})\nHash: {hash}")
                    nop_variable += 1

        return images

class CustomBalancedDataGenerator(Sequence):
    def __init__(self, x_data, y_data, x_hashes, x_landmarks, batch_size, occlusion_probability, masking_function, mismatch, augmentations=None, data_inf=None, label_smoothing=0.1, paths_data=None, matching_amount=0.2, **kwargs):
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
        self.occlusion_layer = RandomOcclusion(occlusion_probability, masking_function, mismatch, matching_amount)

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
            # landmarks_that_should_be = [load_landmark_coordinates(h) for h in batch_x_hashes]

            # Applica il label smoothing
            if self.label_smoothing > 0:
                batch_y = self.apply_label_smoothing(batch_y)

        # Applica il rescale o le trasformazioni per augmentation
        batch_x = self.occlusion_layer(batch_x, np.argmax(batch_y, axis=1), batch_x_landmarks, batch_x_hashes, training=(self.data_inf != 'test'))
        augmented_batch_x = np.zeros_like(batch_x)
        for i in range(len(batch_x)):
            augmented_batch_x[i] = self.augmentations.random_transform(batch_x[i])

        return augmented_batch_x, batch_y

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