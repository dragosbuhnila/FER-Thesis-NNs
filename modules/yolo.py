import mlflow
import numpy as np
from sklearn.metrics import accuracy_score
from ultralytics import YOLO
import torch
import os

from modules.config import ALL_MODELS_PATHS, EMOTIONS, IMAGES_SHAPE, OCCFT_MODELS_FOLDER
from modules.misc import get_timestamp



# INFO on YOLO
# Total modules: 99
# Layers with parameters: 96
#   0 |  | ClassificationModel
#   1 | model | Sequential
#   2 | model.0 | Conv
#   3 | model.0.conv | Conv2d
#   4 | model.0.bn | BatchNorm2d
#   5 | model.0.act | SiLU
#   6 | model.1 | Conv
#   7 | model.1.conv | Conv2d
#   8 | model.1.bn | BatchNorm2d
#   9 | model.2 | C2f
#  10 | model.2.cv1 | Conv
#  11 | model.2.cv1.conv | Conv2d
#  12 | model.2.cv1.bn | BatchNorm2d
#  13 | model.2.cv2 | Conv
#  14 | model.2.cv2.conv | Conv2d
#  15 | model.2.cv2.bn | BatchNorm2d
#  16 | model.2.m | ModuleList
#  17 | model.2.m.0 | Bottleneck
#  18 | model.2.m.0.cv1 | Conv
#  19 | model.2.m.0.cv1.conv | Conv2d
#  20 | model.2.m.0.cv1.bn | BatchNorm2d
#  21 | model.2.m.0.cv2 | Conv
#  22 | model.2.m.0.cv2.conv | Conv2d
#  23 | model.2.m.0.cv2.bn | BatchNorm2d
#  24 | model.3 | Conv
#  25 | model.3.conv | Conv2d
#  26 | model.3.bn | BatchNorm2d
#  27 | model.4 | C2f
#  28 | model.4.cv1 | Conv
#  29 | model.4.cv1.conv | Conv2d
#  30 | model.4.cv1.bn | BatchNorm2d
#  31 | model.4.cv2 | Conv
#  32 | model.4.cv2.conv | Conv2d
#  33 | model.4.cv2.bn | BatchNorm2d
#  34 | model.4.m | ModuleList
#  35 | model.4.m.0 | Bottleneck
#  36 | model.4.m.0.cv1 | Conv
#  37 | model.4.m.0.cv1.conv | Conv2d
#  38 | model.4.m.0.cv1.bn | BatchNorm2d
#  39 | model.4.m.0.cv2 | Conv
#  40 | model.4.m.0.cv2.conv | Conv2d
#  41 | model.4.m.0.cv2.bn | BatchNorm2d
#  42 | model.4.m.1 | Bottleneck
#  43 | model.4.m.1.cv1 | Conv
#  44 | model.4.m.1.cv1.conv | Conv2d
#  45 | model.4.m.1.cv1.bn | BatchNorm2d
#  46 | model.4.m.1.cv2 | Conv
#  47 | model.4.m.1.cv2.conv | Conv2d
#  48 | model.4.m.1.cv2.bn | BatchNorm2d
#  49 | model.5 | Conv
#  50 | model.5.conv | Conv2d
#  51 | model.5.bn | BatchNorm2d
#  52 | model.6 | C2f
#  53 | model.6.cv1 | Conv
#  54 | model.6.cv1.conv | Conv2d
#  55 | model.6.cv1.bn | BatchNorm2d
#  56 | model.6.cv2 | Conv
#  57 | model.6.cv2.conv | Conv2d
#  58 | model.6.cv2.bn | BatchNorm2d
#  59 | model.6.m | ModuleList
#  60 | model.6.m.0 | Bottleneck
#  61 | model.6.m.0.cv1 | Conv
#  62 | model.6.m.0.cv1.conv | Conv2d
#  63 | model.6.m.0.cv1.bn | BatchNorm2d
#  64 | model.6.m.0.cv2 | Conv
#  65 | model.6.m.0.cv2.conv | Conv2d
#  66 | model.6.m.0.cv2.bn | BatchNorm2d
#  67 | model.6.m.1 | Bottleneck
#  68 | model.6.m.1.cv1 | Conv
#  69 | model.6.m.1.cv1.conv | Conv2d
#  70 | model.6.m.1.cv1.bn | BatchNorm2d
#  71 | model.6.m.1.cv2 | Conv
#  72 | model.6.m.1.cv2.conv | Conv2d
#  73 | model.6.m.1.cv2.bn | BatchNorm2d
#  74 | model.7 | Conv
#  75 | model.7.conv | Conv2d
#  76 | model.7.bn | BatchNorm2d
#  77 | model.8 | C2f
#  78 | model.8.cv1 | Conv
#  79 | model.8.cv1.conv | Conv2d
#  80 | model.8.cv1.bn | BatchNorm2d
#  81 | model.8.cv2 | Conv
#  82 | model.8.cv2.conv | Conv2d
#  83 | model.8.cv2.bn | BatchNorm2d
#  84 | model.8.m | ModuleList
#  85 | model.8.m.0 | Bottleneck
#  86 | model.8.m.0.cv1 | Conv
#  87 | model.8.m.0.cv1.conv | Conv2d
#  88 | model.8.m.0.cv1.bn | BatchNorm2d
#  89 | model.8.m.0.cv2 | Conv
#  90 | model.8.m.0.cv2.conv | Conv2d
#  91 | model.8.m.0.cv2.bn | BatchNorm2d
#  92 | model.9 | Classify
#  93 | model.9.conv | Conv
#  94 | model.9.conv.conv | Conv2d
#  95 | model.9.conv.bn | BatchNorm2d
#  96 | model.9.pool | AdaptiveAvgPool2d
#  97 | model.9.drop | Dropout
#  98 | model.9.linear | Linear

FREEZING_MODULES_LAYERS = {
    # train head only unfrozen
    "train_head": 9,
    # last stages
    "last_stages": 7,
    # last two stages
    "last_two_stages": 5,
    # central layers
    "central_layers": 3,
    # most of the model unfrozen
    "most_unfrozen": 2,
}




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


def evaluate_yolo_model_folders_complete(model, test_folder_path):
    """
    Perform inference using a YOLO model on a test folder.

    Parameters:
    - model: YOLO model.
    - test_folder_path: Path to the test folder.

    Returns:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - probabilities: Class probabilities for each image.
    - confidences: Confidence scores for the predictions.
    """
    categories = EMOTIONS
    classes = sorted([folder_name for folder_name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, folder_name))])
    class_to_number = {class_name: index for index, class_name in enumerate(classes)}

    y_true = []
    probabilities = []
    predicted_images_paths = []

    for class_name in classes:
        class_index = class_to_number[class_name]
        class_folder_path = os.path.join(test_folder_path, class_name)
        num_images = len([name for name in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, name))])
        y_true.extend([class_index] * num_images)

        image_files = [name for name in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, name))]
        if not image_files:
            print(f"[WARNING] No images or videos found in {class_folder_path}. Skipping this class.")
            continue

        results = model.predict(class_folder_path)
        probabilities.extend([result.probs.data.cpu().numpy() if isinstance(result.probs.data, torch.Tensor) else result.probs.data for result in results])
        predicted_images_paths.extend([result.path for result in results])

    probabilities = np.array(probabilities)
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    return np.array(y_true), y_pred, probabilities, confidences, predicted_images_paths


def train_model_yolo_training_run(model, trainval_folder,
                                  epochs, batch_size, freezing_module, dropout_rate, patience, learning_rate,
                                  quick_run=False):
    # To use your own dataset with Ultralytics YOLO, ensure it follows the specified directory format required for the classification task, 
    # with separate train, test, and optionally val directories, and subdirectories for each class containing the respective images. 
    # Once your dataset is structured correctly, point # the data argument to your dataset's root directory when initializing the training script. 

    # Training code from notebook:
    # results = model.train(data='/content/drive/MyDrive/Colab Notebooks/HPC/finale/dataset', epochs=300, batch=64, imgsz=128, save_period=3,
    #                       resume = True,
    #                       patience = 30, auto_augment ='autoaugment',
    #                       val=True,save_json=True, plots=True,cache=True,
    #                       mosaic = 0.0, freeze = 5,
    #                       dropout=0.2, lr0=0.001, project='/content/drive/MyDrive/Colab Notebooks/HPC/finale/yolov8n',
    #                       name='yolov8n')#Ricorda di inserire il name corretto per la sottocartella

    # Ensure folders exist
    if not os.path.isdir(trainval_folder):
        raise ValueError(f"Train/val folders not found: {trainval_folder}")

    imgsz = IMAGES_SHAPE[0]
    freezing_layer = FREEZING_MODULES_LAYERS[freezing_module] # apparently freeze parameter = X freezes module X instead of layer number X

    print(f"Starting YOLO training: trainval={trainval_folder}, epochs={epochs}, batch={batch_size}")
    result = None
    try:
        train_kwargs = dict(data=trainval_folder, imgsz=int(imgsz), batch=int(batch_size), auto_augment='autoaugment',
                            epochs=int(epochs), save=True, save_period=3, patience=patience, lr0=learning_rate, optimizer="Adam",
                            val=True, plots=True, cache=True, 
                            mosaic=0.0, freeze=freezing_layer, dropout=dropout_rate) 
        if quick_run:
            # For quick run, we can also use a smaller subset of the data by leveraging the `batch` parameter and early stopping
            train_kwargs['fraction'] = 0.01  
        
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



def save_model_yolo_training_run(model, model_name, test_acc, run_name):
    # For YOLO, we can just save the model using its built-in save method, which creates a .pt file
    base_name = f'{model_name}_occfinetuning'
    base_path = os.path.join(OCCFT_MODELS_FOLDER, f"{base_name}__{run_name}__{test_acc:.4f}")
    os.makedirs(base_path, exist_ok=True)

    try:
        model.save(os.path.join(base_path, f'{base_name}.pt'))
        print(f"YOLO model saved as {os.path.join(base_path, f'{base_name}.pt')}")
    except Exception as e:
        print(f"An error occurred while saving the YOLO model: {e}")


