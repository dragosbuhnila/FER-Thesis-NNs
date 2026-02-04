import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.config import EMOTIONS, GLOBALS
from modules.mask import occlude_batch
from modules.visualize import plot_image



# =============================== SETTINGS AND DEBUG ================================

def refresh_show_flags():
    global SHOW_IMAGES_B4AUG, SHOW_IMAGES_B4AUG_ONLYONCE, SHOW_IMAGES_FINAL, SHOW_IMAGES_FINAL_ONLYONCE
    global SHOW_IMAGES_B4AUG_REMAINING, SHOW_IMAGES_FINAL_REMAINING

    SHOW_IMAGES_B4AUG = GLOBALS.get("DATALOADER_SHOW_IMAGES_B4AUG", False)
    SHOW_IMAGES_B4AUG_ONLYONCE = GLOBALS.get("DATALOADER_SHOW_IMAGES_B4AUG_ONLYONCE", True)
    SHOW_IMAGES_FINAL = GLOBALS.get("DATALOADER_SHOW_IMAGES_FINAL", False)
    SHOW_IMAGES_FINAL_ONLYONCE = GLOBALS.get("DATALOADER_SHOW_IMAGES_FINAL_ONLYONCE", True)

    SHOW_IMAGES_B4AUG_REMAINING = {
        'train': 1 if SHOW_IMAGES_B4AUG_ONLYONCE else float('inf'),
        'valid': 1 if SHOW_IMAGES_B4AUG_ONLYONCE else float('inf'),
        'test' : 1 if SHOW_IMAGES_B4AUG_ONLYONCE else float('inf'),
    }
    SHOW_IMAGES_FINAL_REMAINING = {
        'train': 1 if SHOW_IMAGES_FINAL_ONLYONCE else float('inf'),
        'valid': 1 if SHOW_IMAGES_FINAL_ONLYONCE else float('inf'),
        'test' : 1 if SHOW_IMAGES_FINAL_ONLYONCE else float('inf'),
    }

refresh_show_flags()

def show_dataloader_batch_images(before_or_final: str, split: str, batch_x, batch_y, generator_name, batch_x_hashes=None, mismatched_y=None, positive_or_negative_batch=None):
    if before_or_final.lower() not in ["before", "final"]:
        raise ValueError(f"before_or_final must be either 'before' or 'final', but found is {before_or_final}")
    split = split.lower()
    if split not in ["train", "valid", "test"]:
        raise ValueError(f"split must be either 'train', 'valid', or 'test', instead found {split}")

    dictionary_for_remaining = SHOW_IMAGES_B4AUG_REMAINING if before_or_final.lower() == "before" else SHOW_IMAGES_FINAL_REMAINING

    if dictionary_for_remaining[split] > 0:
        caption = "image(s) before augmentation" if before_or_final.lower() == "before" else "final augmented image(s)"
        print(f"[DEBUG] Showing {caption} for split={split} ({generator_name}).")
        for i in range(len(batch_x)):
            ith_hash = batch_x_hashes[i] if batch_x_hashes is not None else "N/A"
            argmaxed_y = np.argmax(batch_y[i])
            height, width = batch_x[i].shape[0], batch_x[i].shape[1]
            title = f"{split}: {caption} ({height}x{width})\n" + \
                    f"Hash: {ith_hash}. Emotion: {EMOTIONS[argmaxed_y]} (idx {argmaxed_y})"
            if mismatched_y is not None and positive_or_negative_batch is not None:
                mismatched_emotion = EMOTIONS[mismatched_y[i]]
                posneg = "Positive" if positive_or_negative_batch[i] else "Negative"
                title += f"\nMismatched Emotion: {mismatched_emotion} ({posneg})"
            plot_image(batch_x[i], title=title)
        dictionary_for_remaining[split] -= 1

# =============================== END OF SETTINGS AND DEBUG ================================



# =============================== DATA GENERATOR WITH RANDOM OCCLUSION LAYER ================================

def uniform_mismatch(labels, matching_amount, positive_or_negative_batch):
    # Allow for mismatching emotion-landmarks (e.g. putting 'happy' landmarks on a 'sad' image)
    #   > overwrite labels to have a uniform distribution of emotions, except for NEUTRAL (4)

    # 0) Determine amount of matching occlusions
    amt_of_matching_occlusions = int(len(labels) * matching_amount)
    amt_of_mismatching_occlusions = len(labels) - amt_of_matching_occlusions

    # # 1) Select emotions to use for mismatching
    num_emotions = len(EMOTIONS)  # Total number of emotions
    if num_emotions != 7:
        raise ValueError(f"Expected 7 emotions, but got {num_emotions}. Check EMOTIONS mapping.")
    mismatching_emotions_to_apply = [i for i in range(num_emotions) if i != 4] # Exclude NEUTRAL (4) from pool
    mismatching_labels = np.random.choice(mismatching_emotions_to_apply, size=amt_of_mismatching_occlusions)

    # Here there's the choice of keeping always ratios that mirror the test sets (i.e. always having a 20% of the masked images being matched and so on) or allow for the 
    #   non occluded images to substitute any of these, meaning that the ratio may fluctuate inside of the single batch, but should stay kind of consisten throughout the epochs instead.
    # I'll keep this choice of picking the occlusions first (mirroring ratio) and then selecting which images will be occluded or not, since it's the easier approach, 
    #   and should also grant variety to the descent towards optimality.
    # The same philosophy applies to the choice of positive/negative. The distribution within a single batch of every of the 10 kinds of occlusion (nof_occlusion_types = 5 * 2  # 5 occlusion types (each unmatched emotion), each with +/- occlusion)
    #   won't be exaclty uniform since negative/positive isn't really chained to the choice of the mismatching emotion in the following code, but again, 
    #   it should uniform itself throughout the epochs, plus add variability between batches.

    # 2) Find/Decide which indices will have mismatching emotions
    # keep **matching_amount** of labels the same, change the rest
    labels = labels.astype(int)
    indices_randomized = np.arange(len(labels))
    np.random.shuffle(indices_randomized) # This makes indices just a vector with non-repeating random integers, ranging from 0 to len(labels)-1
    mismatch_indices = indices_randomized[amt_of_matching_occlusions:] # Excludes the first *{matching_amount}* of indices
    # mismatch_indices.sort()  # Sort when debugging to check correctness easily

    # 3) Overwrite labels with mismatching ones
    for i, idx in enumerate(mismatch_indices):
        # Get a mismatching label that is different from the original
        original_label = labels[idx]
        mismatched_label = mismatching_labels[i]

        time_spent_in_while_loop = 0
        while mismatched_label == original_label:
            mismatched_label = np.random.choice(mismatching_emotions_to_apply)
            time_spent_in_while_loop += 1

        GLOBALS['MAX_TIME_SPENT_IN_WHILE_LOOP_UNIFORM_MISMATCH'] = max(GLOBALS.get('MAX_TIME_SPENT_IN_WHILE_LOOP_UNIFORM_MISMATCH', 0), time_spent_in_while_loop)

        labels[idx] = mismatched_label

    # 4) Find/Decide which indices (among the mismatching ones) will have positive/negative occlusion
    positive_or_negative_batch[mismatch_indices] = np.random.randint(0, 2, size=len(mismatch_indices))

    return labels, positive_or_negative_batch

def specific_mismatch(labels, emotion_name):
    if emotion_name.upper() == 'NEUTRAL':
        raise ValueError("NEUTRAL emotion cannot be used for specific_mismatch, as it is not allowed to mismatch to NEUTRAL.")
    if emotion_name not in EMOTIONS:
        raise ValueError(f"emotion_name must be one of {', '.join(EMOTIONS)}, but found '{emotion_name}'")

    target_emotion_idx = EMOTIONS.index(emotion_name) # Get index of the target emotion

    # 1) Overwrite all labels with the specific mismatching one
    labels = np.array([target_emotion_idx] * len(labels))

    return labels


class RandomOcclusion(keras.layers.Layer):
    def __init__(self, occlusion_probability: float, masking_function_name: str, mismatch: str, matching_amount: float, pos_or_neg: str = None, **kwargs):
        super().__init__(**kwargs)
        self.occlusion_probability = occlusion_probability
        self.masking_function_name = masking_function_name
        self.mismatch = mismatch
        self.matching_amount = matching_amount
        self.pos_or_neg = pos_or_neg
    def call(self, images, labels, image_landmarks, image_hashes):        
        # 1) Preparing parameters
        images = images.numpy()

        batch_size = images.shape[0]
        amt_to_occlude = int(batch_size * self.occlusion_probability)
        
        if self.pos_or_neg is None:
            positive_or_negative_batch = np.ones_like(labels) # Matching indices will always be positive occlusion (1)
        elif self.pos_or_neg.lower() == 'positive':
            positive_or_negative_batch = np.ones_like(labels) # All occlusions will be positive (1)
        elif self.pos_or_neg.lower() == 'negative':
            positive_or_negative_batch = np.zeros_like(labels) # All occlusions will be negative (0)

        if amt_to_occlude == 0:
            return images, labels, positive_or_negative_batch

        # 2) Prepare for mismatched occlusion (if needed)
        if self.mismatch.lower() == 'uniform':
            # uniform mismatch across all emotions except NEUTRAL (4). 
            labels, positive_or_negative_batch = uniform_mismatch(labels, self.matching_amount, positive_or_negative_batch) 
        elif self.mismatch.upper() in EMOTIONS:
            labels = specific_mismatch(labels, self.mismatch.upper())
        elif self.mismatch.lower() != 'none' and self.mismatch.lower() != 'no':
            raise ValueError(f"mismatch must be either 'uniform', 'none'/'no', or an emotion name ({', '.join(EMOTIONS)}), but found '{self.mismatch}'")

        # 3) Prepare for occlusion
        # > Decide which images to occlude
        indices_randomized = np.arange(len(labels))
        np.random.shuffle(indices_randomized) # This makes indices just a vector with non-repeating random integers, ranging from 0 to len(labels)-1
        occlude_indices = indices_randomized[:amt_to_occlude]

        # > Sub-batch to occlude
        images_sub = images[occlude_indices]
        labels_sub = labels[occlude_indices]
        landmarks_sub = image_landmarks[occlude_indices]
        hashes_sub = image_hashes[occlude_indices]
        posneg_sub = positive_or_negative_batch[occlude_indices]

        # 4) Apply occlusion only to sub-batch
        occluded_sub = occlude_batch(images_sub, labels_sub, landmarks_sub, hashes_sub, posneg_sub, self.masking_function_name)
        if tf.is_tensor(occluded_sub):
            occluded_sub = occluded_sub.numpy()
        elif isinstance(occluded_sub, list):
            occluded_sub = np.array(occluded_sub)
        else:
            raise TypeError("Occluded images must be either a tf.Tensor or a list, there must have been an issue.")
        
        # 5) Reinsert occluded sub-batch into original batch
        images[occlude_indices] = occluded_sub

        return images, labels, positive_or_negative_batch

class OnlineOcclusionGenerator(Sequence):
    def __init__(self, x_data, y_data, x_hashes, x_landmarks, batch_size, occlusion_probability, masking_function_name, mismatch, augmentations=None, data_inf=None, label_smoothing=0.1, paths_data=None, matching_amount=0.2, pos_or_neg=None, dont_rebalance_trainval=False, **kwargs):
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
        self.occlusion_layer = RandomOcclusion(occlusion_probability, masking_function_name, mismatch, matching_amount, pos_or_neg=pos_or_neg)

        self.dont_rebalance_trainval = dont_rebalance_trainval

        if data_inf in ['train', 'valid']:
            #print(y_data)
            # 1) Augment
            self.augmentations = ImageDataGenerator(**augmentations)

            # 2) Class balancing
            self.classes = np.unique(np.argmax(y_data, axis=1))  # Ricaviamo le classi dai dati one-hot encoded
            self.class_indices = {cls: np.where(np.argmax(y_data, axis=1) == cls)[0] for cls in self.classes}
            self.num_classes = len(self.classes)
            self.samples_per_class = max(1, self.batch_size // self.num_classes) # e.g. 7 classes and bs=64: 64/7=9 samples per class
            self.class_pointers = {cls: 0 for cls in self.classes} # Coda ciclica per le classi minoritarie
        elif data_inf == 'test':
            self.augmentations = ImageDataGenerator(**(augmentations or {}))
        
        self.index = 0
        self.on_epoch_end() # Shuffles data by shuffling indices (only in train/valid)
        print(f"Generator initialized: {data_inf} mode")

    def __len__(self):
        # Note that, since we perform class balancing in train/valid mode, the number of batches per epoch is approximate.
        #   i.e. each epoch may not see all samples exactly once AND it may not see all the samples.
        #           (why? Because it will see unpopular classes a lot of times, taking slots in the batches, so the popular
        #                   batches like happy will run out of time before the epoch ends because of reacing __len__(), 
        #                   because len is defined in a way that does not take into account the rebalancing)
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
        if self.data_inf == 'test' or self.dont_rebalance_trainval==True:
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
        batch_x, mismatched_y, positive_or_negative_batch = self.occlusion_layer(batch_x, np.argmax(batch_y, axis=1), batch_x_landmarks, batch_x_hashes)
        # print(f"[DEBUG] SHOW_IMAGES_B4AUG={SHOW_IMAGES_B4AUG}, SHOW_IMAGES_FINAL={SHOW_IMAGES_FINAL} for data_inf={self.data_inf} (CustomBalancedDataGenerator).")
        if SHOW_IMAGES_B4AUG:
            show_dataloader_batch_images(before_or_final="before", split=self.data_inf, batch_x=batch_x, batch_y=batch_y, generator_name="CustomBalancedDataGenerator", batch_x_hashes=batch_x_hashes, mismatched_y=mismatched_y, positive_or_negative_batch=positive_or_negative_batch)

        for i in range(len(batch_x)):
            batch_x[i] = self.augmentations.random_transform(batch_x[i])
        if SHOW_IMAGES_FINAL:
            show_dataloader_batch_images(before_or_final="final", split=self.data_inf, batch_x=batch_x, batch_y=batch_y, generator_name="CustomBalancedDataGenerator", batch_x_hashes=batch_x_hashes, mismatched_y=mismatched_y, positive_or_negative_batch=positive_or_negative_batch)

        return batch_x, batch_y

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

class OfflineOcclusionGenerator(Sequence):
    def __init__(self, split, 
                 x_data, y_data, x_hashes, occlusion_indexer, types_of_occlusion,
                 batch_size, augmentations=None, label_smoothing=0.1,
                 occlusion_probability=0.2, masking_function_name="lines", # matching_amount=None, # Not implemented for now, just choose uniformly
                 dont_rebalance_trainval=False, **kwargs):
        super().__init__(**kwargs)

        if split not in ['train', 'valid']: # Occlusion is only applied during training/validation. Test set is already occluded.
            raise ValueError(f"split must be 'train' or 'valid', but is '{split}'")
        self.split = split

        self.x_data = x_data
        self.y_data = y_data
        self.x_hashes = x_hashes
        self.indices = np.arange(len(x_data))
        
        self.batch_size = batch_size
        self.augmentations = ImageDataGenerator(**augmentations)
        self.label_smoothing = label_smoothing

        self.occlusion_probability = occlusion_probability
        self.masking_function_name = masking_function_name

        self.rng = np.random.default_rng()

        self.dont_rebalance_trainval = dont_rebalance_trainval
        if not dont_rebalance_trainval:
            self.classes = np.unique(np.argmax(y_data, axis=1))  # Ricaviamo le classi dai dati one-hot encoded
            self.class_indices = {cls: np.where(np.argmax(y_data, axis=1) == cls)[0] for cls in self.classes}
            self.num_classes = len(self.classes)
            self.samples_per_class = max(1, self.batch_size // self.num_classes) # e.g. 7 classes and bs=64: 64/7=9 samples per class
            self.class_pointers = {cls: 0 for cls in self.classes} # Coda ciclica per le classi minoritarie
        
        self.index = 0
        self.on_epoch_end() # Shuffles data by shuffling indices (only in train/valid)
        print(f"Generator initialized: {split} mode")

    def __len__(self):
        # Note that, since we perform class balancing in train/valid mode, the number of batches per epoch is approximate.
        #   i.e. each epoch may not see all samples exactly once AND it may not see all the samples.
        #           (why? Because it will see unpopular classes a lot of times, taking slots in the batches, so the popular
        #                   batches like happy will run out of time before the epoch ends because of reacing __len__(), 
        #                   because len is defined in a way that does not take into account the rebalancing)
        return int(np.ceil(len(self.x_data) / self.batch_size))
    
    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        batch = self.__getitem__(self.index)
        self.index += 1
        return batch

    def __iter__(self):
        self.index = 0
        return self
    
    def __getitem__(self, index):
        if self.dont_rebalance_trainval==True:
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.x_data))

            batch_x, batch_y, batch_x_hashes = self.x_data[start_idx:end_idx], self.y_data[start_idx:end_idx], self.x_hashes[start_idx:end_idx]
        else:
            batch_x, batch_y, batch_x_hashes = [], [], []

            # Balance batches by going through each class, taking samples_per_class samples from each, and shuffling each class independently.
            # This is done by leveraging 
            #       > the class_indices dictionary, that yields all the indices given a class,
            #       > and a class_pointer dictionary, that keeps track of where we are in the circular queue for each class.
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
                except Exception as exc:
                    raise

                # Aggiorna il puntatore per la classe
                self.class_pointers[cls] += len(selected_indices)

                # Se abbiamo esaurito i dati per la classe, fai uno shuffle e riparti
                if self.class_pointers[cls] >= len(cls_indices):
                    self.class_pointers[cls] = 0
                    np.random.shuffle(cls_indices)  # Shuffle della classe
                    self.class_indices[cls] = cls_indices

            batch_x, batch_y, batch_x_hashes = np.array(batch_x), np.array(batch_y), np.array(batch_x_hashes)
            batch_x, batch_y, batch_x_hashes = shuffle(batch_x, batch_y, batch_x_hashes)

            if self.label_smoothing > 0:
                batch_y = self.apply_label_smoothing(batch_y)

        # Augmentations (including occlusion)
        occlude_or_not = np.random.rand(len(batch_x)) < self.occlusion_probability
        occlusion_type = np.random.choice(list(self.types_of_occlusion), size=len(batch_x))
        for i in range(len(batch_x)):
            if occlude_or_not[i]:
                batch_x[i] = self.occlusion_indexer[batch_x_hashes[i]][occlusion_type[i]]

        if SHOW_IMAGES_B4AUG:
            show_dataloader_batch_images(before_or_final="before", split=self.split, batch_x=batch_x, batch_y=batch_y, generator_name="OfflineOcclusionGenerator", batch_x_hashes=batch_x_hashes, mismatched_y=mismatched_y, positive_or_negative_batch=positive_or_negative_batch)
        for i in range(len(batch_x)):
            batch_x[i] = self.augmentations.random_transform(batch_x[i])
        if SHOW_IMAGES_FINAL:
            show_dataloader_batch_images(before_or_final="final", split=self.split, batch_x=batch_x, batch_y=batch_y, generator_name="OfflineOcclusionGenerator", batch_x_hashes=batch_x_hashes, mismatched_y=mismatched_y, positive_or_negative_batch=positive_or_negative_batch)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.split != 'test':
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


class OldCustomBalancedDataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size, augmentations=None, data_inf=None, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.data_inf = data_inf
        self.label_smoothing = label_smoothing
        self.indices = np.arange(len(x_data))

        # Se siamo in 'train' o 'valid', impostiamo le augmentation e il bilanciamento
        if data_inf in ['train', 'valid']:
            #print(y_data)
            self.augmentations = ImageDataGenerator(**augmentations)
            self.classes = np.unique(np.argmax(y_data, axis=1))  # Ricaviamo le classi dai dati one-hot encoded
            self.class_indices = {cls: np.where(np.argmax(y_data, axis=1) == cls)[0] for cls in self.classes}
            self.num_classes = len(self.classes)
            self.samples_per_class = max(1, self.batch_size // self.num_classes)

            # Coda ciclica per le classi minoritarie
            self.class_pointers = {cls: 0 for cls in self.classes}

        # Se siamo in 'test', usiamo solo rescale e nessuna augmentation o bilanciamento
        elif data_inf == 'test':
            self.augmentations = ImageDataGenerator(**(augmentations or {}))

        self.on_epoch_end()
        print(f"Generator initialized: {data_inf} mode")

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        if self.data_inf == 'test':
            # Per il test set, usiamo semplicemente gli indici
            start_idx = index * self.batch_size
            end_idx = min((index + 1) * self.batch_size, len(self.x_data))
            batch_x = self.x_data[start_idx:end_idx]
            batch_y = self.y_data[start_idx:end_idx]
        else:
            # Per train/valid, selezioniamo batch bilanciati
            batch_x, batch_y = [], []
            for cls in self.classes:
                cls_indices = self.class_indices[cls]
                cls_pointer = self.class_pointers[cls]

                # Seleziona i dati dalla coda ciclica
                selected_indices = cls_indices[cls_pointer:cls_pointer + self.samples_per_class]
                batch_x.extend(self.x_data[selected_indices])
                batch_y.extend(self.y_data[selected_indices])

                # Aggiorna il puntatore per la classe
                self.class_pointers[cls] += len(selected_indices)

                # Se abbiamo esaurito i dati per la classe, fai uno shuffle e riparti
                if self.class_pointers[cls] >= len(cls_indices):
                    self.class_pointers[cls] = 0
                    np.random.shuffle(cls_indices)  # Shuffle della classe
                    self.class_indices[cls] = cls_indices

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            batch_x, batch_y = shuffle(batch_x, batch_y)

            # Applica il label smoothing
            if self.label_smoothing > 0:
                batch_y = self.apply_label_smoothing(batch_y)


        if SHOW_IMAGES_B4AUG:
            show_dataloader_batch_images(before_or_final="before", split=self.data_inf, batch_x=batch_x, batch_y=batch_y, generator_name="OldCustomBalancedDataGenerator")

        # Applica il rescale o le trasformazioni per augmentation
        for i in range(len(batch_x)):
            batch_x[i] = self.augmentations.random_transform(batch_x[i])
        if SHOW_IMAGES_FINAL:
            show_dataloader_batch_images(before_or_final="final", split=self.data_inf, batch_x=batch_x, batch_y=batch_y, generator_name="OldCustomBalancedDataGenerator")

        # train_generator, valid_generator, test_generator, initial_bias
        return batch_x, batch_y

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