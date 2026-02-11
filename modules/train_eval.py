import os
import mlflow
import tensorflow as tf
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from modules.config import EMOTIONS, OCCFT_MODELS_FOLDER
from modules.misc import get_timestamp


def evaluate_yolo_model_folders(model, test_folder_path, debug=False):
    # test_folder_path should be something like C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\data\datasets\bosphorus_test_finale
    categories = EMOTIONS

    classes = [folder_name for folder_name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, folder_name))]
    classes.sort()
    class_to_number = {class_name: index for index, class_name in enumerate(classes)}

    true_labels = []

    for class_name in classes:
        class_index = class_to_number[class_name]
        class_folder_path = os.path.join(test_folder_path, class_name)
        num_images = len([name for name in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, name))])
        true_labels.extend([class_index] * num_images)

    if debug:
        print("Class to Number Mapping:", class_to_number)
        print("Classes (Alphabetical Order):", classes)
        print("True Labels (Numerical):", true_labels)

    pred_labels = []

    for category in categories:
        # Eseguire la predizione per ogni cartella di immagini
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(f'{os.path.join(test_folder_path, category)}', device=device)

        # Estrarre le probabilità di classe per ogni risultato e aggiungerle alla lista
        pred_labels.extend([result.probs.top1 for result in results])
    
    # Calcolo dell'accuratezza
    accuracy = accuracy_score(true_labels, pred_labels)

    # 5) Return None for loss (not computed), and accuracy
    return None, accuracy

def evaluate_yolo_model_testgen(model, test_generator, debug=False):
    """Evaluate an Ultralytics YOLO model manually over a Keras-style generator.

    Returns (loss, accuracy). Only accuracy is computed; loss is returned
    as None (placeholder) since you only care about accuracy.
    """
    # 0) Ensure generator iterator state is reset
    try:
        iter(test_generator)
    except Exception:
        pass
    
    pred_labels = []
    true_labels = []

    for batch in test_generator:
        # 1) generator yields (X_batch, y_batch)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch = batch[0], batch[1]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")

        # 2a) Convert one-hot y_batch to integer labels
        y_int = np.argmax(y_batch, axis=1)

        # 2b) Listify X_batch
        # X_as_list = [X_for_model[i] for i in range(X_for_model.shape[0])]
        X_as_list = [X_batch[i] for i in range(X_batch.shape[0])]

        # 3a) Run prediction 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(source=X_as_list, device=device)

        # 3b) Save predictions and labels
        pred_labels.extend([result.probs.top1 for result in results])
        true_labels.extend(y_int.tolist())

    # 4) Compute accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # 5) Return None for loss (not computed), and accuracy
    return None, accuracy

def evaluate_model(model, model_name, test_generator, yolo_test_folder_path=None, debug=False):
    if yolo_test_folder_path is not None and test_generator is not None:
        raise ValueError("Provide either yolo_test_folder_path or test_generator, not both.")

    if "yolo" in model_name:
        if yolo_test_folder_path:
            # raise ValueError("I don't want this to be used anymore. Don't provide yolo_test_folder_path, only data_generators.")
            print("Evaluating YOLO model using folder structure instead of test_generator...")
            test_loss, test_acc = evaluate_yolo_model_folders(model, yolo_test_folder_path, debug=debug)
        else:
            test_loss, test_acc = evaluate_yolo_model_testgen(model, test_generator, debug=debug)
    else:
        test_loss, test_acc = model.evaluate(test_generator)

    if test_loss is None:
        test_loss = -1.0 
    return test_loss, test_acc



def addestra_modello(model, train_generator, valid_generator, test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True, mode = 'max')
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    history = model.fit(train_generator, epochs=TRAIN_EPOCH, validation_data=valid_generator, verbose=1,
                        callbacks=[early_stopping_callback, learning_rate_callback])
    
    #  # Loggare l'accuratezza del training e della validazione su Neptune
    # for epoch in range(len(history.history['categorical_accuracy'])):
    #     run[f"{model_name}/finetuning/training/accuracy"].log(history.history['categorical_accuracy'][epoch])
    #     run[f"{model_name}/finetuning/validation/accuracy"].log(history.history['val_categorical_accuracy'][epoch])
    #     run[f"{model_name}/finetuning/training/loss"].log(history.history['loss'][epoch])
    #     run[f"{model_name}/finetuning/validation/loss"].log(history.history['val_loss'][epoch])

    # Log to ml flow instead
    for epoch in range(len(history.history['categorical_accuracy'])):
        mlflow.log_metric("training_accuracy", history.history['categorical_accuracy'][epoch], step=epoch)
        mlflow.log_metric("validation_accuracy", history.history['val_categorical_accuracy'][epoch], step=epoch)
        mlflow.log_metric("training_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("validation_loss", history.history['val_loss'][epoch], step=epoch)
    
    return history

def valuta_modello(model, test_generator, run, model_name):
    test_loss, test_acc = model.evaluate(test_generator)

    # # Loggare l'accuratezza del test su Neptune
    # run[f"{model_name}/finetuning/test/loss"].log(test_loss)
    # run[f"{model_name}/finetuning/test/accuracy"].log(test_acc)

    # Log on ml flow instead
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)

    return test_loss, test_acc

def salva_modello(model, run, model_name):
    base_name = f'{model_name}_occfinetuning'
    base_path = os.path.join(OCCFT_MODELS_FOLDER, f"{base_name}_{get_timestamp()}")
    
    try:
        # Salva il modello in formato TensorFlow
        tf_model_path = os.path.join(base_path, base_name)
        model.save(tf_model_path, save_format='tf')
        print(f"Model saved as {tf_model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model in TensorFlow format: {e}")

    try:
        # Salva il modello in formato HDF5
        h5_model_path = os.path.join(base_path, f'{base_name}.h5')
        model.save(h5_model_path, save_format='h5')
        print(f"Model saved as {h5_model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model in HDF5 format: {e}")

    try:
        # Salva i pesi del modello
        weights_path = os.path.join(base_path, f'{base_name}_weights.h5')
        model.save_weights(weights_path)
        weights_path = os.path.join(base_path,f'{base_name}.weights.h5')
        model.save_weights(weights_path)
        print(f"Model weights saved as {weights_path}")
    except Exception as e:
        print(f"An error occurred while saving the model weights: {e}")

    try:
        keras_model_path = os.path.join(base_path, f'{base_name}.keras')
        # Salva il modello in formato Keras
        model.save(keras_model_path, save_format='keras')
        print(f"Model saved as {base_name}.keras")
    except Exception as e:
        print(f"An error occurred while saving the model in Keras format: {e}")

    # try:
    #     # Carica i file su Neptune
    #     run[f"{model_name}/saved_model"].upload(tf_model_path)
    #     run[f"{model_name}/saved_weights"].upload(weights_path)
    # except Exception as e:
    #     print(f"An error occurred while uploading the model to Neptune: {e}")

    # Use mlflow to log the model instead
    try:
        try:
            sample_input = np.random.random((1, IMAGES_SHAPE[0], IMAGES_SHAPE[1], IMAGES_SHAPE[2]))
            sample_output = model.predict(sample_input)
            signature = infer_signature(sample_input, sample_output)
        except Exception as e:
            print(f"An error occurred while inferring the model signature: {e}")
            signature = None

        if signature is not None:
            mlflow.tensorflow.log_model(model, name=f"{base_name}_mlflow_model", signature=signature)
            print(f"Model logged to MLflow with name: {base_name}_mlflow_model and signature.")
        else:
            mlflow.tensorflow.log_model(model, name=f"{base_name}_mlflow_model")
            print(f"Model logged to MLflow with name: {base_name}_mlflow_model without signature.")
    except Exception as e:
        print(f"An error occurred while logging the model to MLflow: {e}")