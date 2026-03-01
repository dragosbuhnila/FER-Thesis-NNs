import mlflow
from sklearn.base import accuracy_score
from ultralytics import YOLO
import torch
import os

from modules.config import ALL_MODELS_PATHS, EMOTIONS, IMAGES_SHAPE, OCCFT_MODELS_FOLDER
from modules.misc import get_timestamp



def load_yolo_model(model_name, model_path_subset=ALL_MODELS_PATHS, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if debug:
        print(f"Loading YOLO model on device: {device}")
    print(f"Loading YOLO model: {model_name} from {model_path_subset[model_name]}")
    return YOLO(model_path_subset[model_name]).to(device)

    
def evaluate_yolo_model_folders(model, test_folder_path, debug=False):
    # test_folder_path should be something like C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\data\datasets\bosphorus_test_finale
    categories = EMOTIONS

    classes = [folder_name for folder_name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, folder_name))]
    classes.sort()
    class_to_number = {class_name: index for index, class_name in enumerate(classes)}

    for category, class_name in zip(categories, classes):
        if category != class_name:
            raise ValueError(f"Category '{category}' does not match class folder '{class_name}'. Please ensure the test folder is organized with subfolders named after the categories in EMOTIONS.")

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


def train_model_yolo_training_run(model, train_folder, val_folder,
                                  epochs, batch_size, learning_rate):

    # Ensure folders exist
    if not os.path.isdir(train_folder) or not os.path.isdir(val_folder):
        raise ValueError(f"Train/val folders not found: {train_folder}, {val_folder}")

    data_spec = {
        "train": train_folder,
        "val": val_folder,
        "nc": len(EMOTIONS),
        "names": {i: name for i, name in enumerate(EMOTIONS)}
    }

    imgsz = IMAGES_SHAPE[0]

    print(f"Starting YOLO training: train={train_folder}, val={val_folder}, epochs={epochs}, batch={batch_size}, lr0={learning_rate}")
    result = None
    try:
        train_kwargs = dict(data=data_spec, epochs=int(epochs), batch=int(batch_size))
        train_kwargs['imgsz'] = imgsz
        train_kwargs['lr0'] = learning_rate

        result = model.train(**train_kwargs)
        print("YOLO training finished.")
    except Exception as e:
        print(f"Error while training YOLO model: {e}")
        raise

    # Try to extract and log some useful metrics to MLflow
    try:
        # `result` from ultralytics might include `metrics` or `yaml` summary depending on version
        if result is not None:
            # history-like attribute: look for `metrics` or `history`
            metrics = getattr(result, "metrics", None) or getattr(result, "history", None) or {}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
            # if result has a summary or best fitness, try to log it
            best_fitness = getattr(result, "best_fitness", None)
            if best_fitness is not None:
                mlflow.log_metric("best_fitness", float(best_fitness))
    except Exception as e:
        print(f"Could not log YOLO training metrics to MLflow: {e}")

    return result


def evaluate_model_yolo_training_run(model, test_folder_path, debug=False):

    # Evaluate on provided test folder if present
    try:
        if test_folder_path and os.path.isdir(test_folder_path):
            print("Evaluating trained YOLO model on test folder...")
            _, test_acc = evaluate_yolo_model_folders(model, test_folder_path, debug=debug)
            mlflow.log_metric("test_accuracy", float(test_acc))
        else:
            test_acc = -1.0
    except Exception as e:
        print(f"Error during YOLO evaluation: {e}")
        test_acc = -1.0

    return test_acc



def save_model_yolo_training_run(model, model_name, test_acc):
    # For YOLO, we can just save the model using its built-in save method, which creates a .pt file
    base_name = f'{model_name}_occfinetuning'
    base_path = os.path.join(OCCFT_MODELS_FOLDER, f"{base_name}__{get_timestamp()}__{test_acc:.4f}")
    os.makedirs(base_path, exist_ok=True)

    try:
        model.save(os.path.join(base_path, f'{base_name}.pt'))
        print(f"YOLO model saved as {os.path.join(base_path, f'{base_name}.pt')}")
    except Exception as e:
        print(f"An error occurred while saving the YOLO model: {e}")

    # Save with mlflow too
    try:
        mlflow.pytorch.log_model(model, name=f"{base_name}_mlflow_model")
        print(f"YOLO model logged to MLflow with name: {base_name}_mlflow_model")
    except Exception as e:
        print(f"An error occurred while logging the YOLO model to MLflow: {e}")

