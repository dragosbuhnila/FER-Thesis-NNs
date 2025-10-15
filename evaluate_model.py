import h5py
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.config import ADELE_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, EMOTIONS, ALL_MODELS_PATHS


class CustomBalancedDataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size, augmentations=None, data_inf=None, label_smoothing=0.1,paths_data=None, **kwargs):
        super().__init__(**kwargs)
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.data_inf = data_inf
        self.label_smoothing = label_smoothing
        self.indices = np.arange(len(x_data))
        self.paths_data = paths_data


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
        self.index = 0
        self.on_epoch_end()
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
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            # Se hai i path
            if self.paths_data is not None:
                batch_paths = self.paths_data[start_idx:end_idx]
            else:
                batch_paths = None
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
            batch_paths = None



        # Applica il rescale o le trasformazioni per augmentation
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
        
def categorical_focal_loss(alpha=0.25, gamma=2.0):
        """
        Implementazione della categorical focal loss per etichette one-hot.

        Args:
            alpha (float): Ponderazione degli esempi positivi.
            gamma (float): Esponente che controlla il peso degli esempi ben classificati.

        Returns:
            Callable: Funzione di perdita focal loss.
        """
        def loss(y_true, y_pred):
            # Garantisce che le predizioni siano comprese tra 0 e 1
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

            # Calcolo della cross-entropy
            ce = -y_true * tf.math.log(y_pred)

            # Calcolo del fattore modulatorio focal
            modulating_factor = tf.pow(1.0 - y_pred, gamma)

            # Calcolo della focal loss
            focal_loss = alpha * modulating_factor * ce

            # Ritorno della perdita media per batch
            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

        return loss


def load_model(model_name, model_path_subset):
    custom_objects = {'loss': categorical_focal_loss()}

    if not model_name in model_path_subset.keys():
        raise ValueError(f"Model name '{model_name}' not found in the provided model path subset.")

    if "efficientnet" in model_name:
        return None

    if "yolo" in model_name:
        return None
    
    # For other models. Probably: resnet, vgg, inception, convnext, pattlite
    with keras.utils.custom_object_scope(custom_objects):
        # One of these three should work:
        # model = keras.layers.TFSMLayer(model_path_subset[model_name], call_endpoint='serve')
        # model = keras.layers.TFSMLayer(model_path_subset[model_name], call_endpoint='serving_default')
        model = keras.models.load_model(model_path_subset[model_name])
        return model
    
def load_data_generator(path, title):
    def load_data_and_labels(file_path, info):
        class_names = None
        with h5py.File(file_path, 'r') as f:
            if info == 'train':
                raise NotImplementedError("Training data loading not implemented.")
            elif info == 'test':
                x = np.array(f['X_test'])
                y = np.array(f['y_test'])
                # Leggiamo anche i path se esistono
                if 'paths' in f:
                    # Se 'paths' è un dataset di stringhe a lunghezza variabile
                    # con h5py.string_dtype, possiamo leggerlo direttamente:
                    paths_data = f['paths'][...]  # np array di stringhe
                else:
                    paths_data = None
                return x, y, class_names, paths_data
            else:
                raise ValueError(f"Info must be 'train' or 'test', but is '{info}'")

    NUM_CLASSES = len(EMOTIONS)
    X_test, y_test, class_names, test_paths = load_data_and_labels(path, title)
    y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)
    test_augmentations = {}

    data_generator = CustomBalancedDataGenerator(
        x_data=X_test,
        y_data=y_test_one_hot,
        batch_size=64,
        augmentations=test_augmentations,
        data_inf='test',
        label_smoothing=0,
        paths_data=test_paths  # Nuovo parametro
    )

    return data_generator

def evaluate_model(model, model_name, test_generator):
    if "efficientnet" in model_name:
        return None

    if "yolo" in model_name:
        return None
    
    # For other models. Probably: resnet, vgg, inception, convnext, pattlite
    test_loss, test_acc = model.evaluate(test_generator)
    return test_loss, test_acc

def generate_h5_from_images():
    class_names = sorted(os.listdir(OCCLUDED_TEST_SET_PATH))
    paths = []
    X_test = []
    y_test = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(OCCLUDED_TEST_SET_PATH, class_name)
        image_files = sorted(os.listdir(class_folder))
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            # Load PNG image as RGB NumPy array and resize to (128, 128, 3)
            image = np.array(Image.open(image_path).convert('RGB').resize((128, 128)))
            X_test.append(image)
            y_test.append(class_idx)  # Store class index instead of name
            paths.append(image_path)

            # Also save the images to OCCLUDED_TEST_SET_RESIZED_PATH
            save_folder = os.path.join(OCCLUDED_TEST_SET_RESIZED_PATH, class_name)
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(save_folder, image_file))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 3) Save new h5
    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, "w") as f:
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)  # Now integers
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))  # Save as bytes
        f.create_dataset("paths", data=np.array(paths).astype('S'))  # Save as bytes
    print(f"Saved {X_test.shape[0]} images to {OCCLUDED_TEST_SET_H5_PATH}")

    # 4) Check saved h5 to have 350 images, 7 classes, 50 images per class
    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, "r") as f:
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


if __name__ == "__main__":
    # 1) Load the test set
    # generate_h5_from_images() # Generate from scratch at least the first time running on a different location
    test_set_path = ADELE_TEST_SET_H5_PATH
    test_generator = load_data_generator(test_set_path, 'test')

    # 1) ========================================================


    # 2) Run the evaluations on the test set
    models_names = ["resnet_finetuning", "pattlite_finetuning", "vgg19_finetuning", "inceptionv3_finetuning", "convnext_finetuning"]
    models_results = {name: {"test_loss": None, "test_acc": None} for name in models_names}

    for model_name in models_names:
        print("======================================")
        print(f"Evaluating model: {model_name}")

        model = load_model(model_name, ALL_MODELS_PATHS)
        if model is None:
            print("Model loading not implemented for this model type.")
            continue
        else:
            test_loss, test_acc = evaluate_model(model, model_name, test_generator)
            models_results[model_name]["test_loss"] = test_loss
            models_results[model_name]["test_acc"] = test_acc

    print("======================================")
    # 2) ========================================================

    print("Final evaluation results on occluded test set:")
    for model_name, results in models_results.items():
        print(f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}")

    # train
    