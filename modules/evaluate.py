import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

from modules.config import ALL_MODELS_PATHS, EMOTIONS



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from modules.config import EMOTIONS, SAVED_IMAGES_PATH
from modules.misc import get_timestamp, create_placeholder_image



# ================================================================================================================
# ============================ Visualization =====================================================================
# ================================================================================================================

def save_or_show_figure(fig, save_path, save_instead_of_show):
    """Save the figure to a file or display it."""
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Saved figure to {save_path}")
    if not save_instead_of_show:
        plt.show()
    plt.close(fig)


def plot_images_with_predictions(axes, chunk, test_generator, y_pred, y_true, confidences, class_names_fixed, placeholder_array):
    """Plot images with predictions and fill remaining slots with placeholders."""
    for i, idx in enumerate(chunk):
        img_data = test_generator.x_data[idx].astype('uint8')
        pil_img = Image.fromarray(img_data).resize((224, 224), Image.BILINEAR)
        resized_img = np.array(pil_img)

        axes[i].imshow(resized_img)
        axes[i].set_title(
            f"Pred: {class_names_fixed[y_pred[idx]]}\n"
            f"Conf: {confidences[idx]:.2f}\n"
            f"True: {class_names_fixed[y_true[idx]]}"
        )
        axes[i].axis("off")

    for j in range(len(chunk), len(axes)):
        axes[j].imshow(placeholder_array)
        axes[j].axis("off")


def visualize_high_confidence_errors(sorted_indices, test_generator, y_pred, y_true, confidences, class_names_fixed, save_dir, save_instead_of_show):
    """Visualize high-confidence errors in batches."""
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 18
    placeholder_array = create_placeholder_image()

    for fig_idx, start in enumerate(range(0, len(sorted_indices), batch_size), start=1):
        end = start + batch_size
        chunk = sorted_indices[start:end]

        fig, axes = plt.subplots(6, 3, figsize=(9, 18))
        axes = axes.flatten()

        plot_images_with_predictions(axes, chunk, test_generator, y_pred, y_true, confidences, class_names_fixed, placeholder_array)

        save_path = os.path.join(save_dir, f"high_conf_error_{fig_idx}.png")
        save_or_show_figure(fig, save_path, save_instead_of_show)


def visualize_uncertain_predictions(uncertain_indices, test_generator, y_pred, y_true, probabilities, class_names_fixed, save_dir, save_instead_of_show):
    """Visualize uncertain predictions in batches."""
    os.makedirs(save_dir, exist_ok=True)

    batch_size = 18
    placeholder_array = create_placeholder_image()

    for fig_idx, start in enumerate(range(0, len(uncertain_indices), batch_size), start=1):
        end = start + batch_size
        chunk = uncertain_indices[start:end]

        fig, axes = plt.subplots(6, 3, figsize=(9, 18))
        axes = axes.flatten()

        for i, idx in enumerate(chunk):
            img_data = test_generator.x_data[idx].astype('uint8')
            pil_img = Image.fromarray(img_data).resize((224, 224), Image.BILINEAR)
            resized_img = np.array(pil_img)

            top_2_indices = np.argsort(probabilities[idx])[-2:][::-1]
            top_2_classes = [class_names_fixed[i] for i in top_2_indices]
            top_2_probs = probabilities[idx][top_2_indices]

            axes[i].imshow(resized_img)
            axes[i].set_title(
                f"Pred: {top_2_classes[0]} ({top_2_probs[0]:.2f})\n"
                f"Sec: {top_2_classes[1]} ({top_2_probs[1]:.2f})\n"
                f"True: {class_names_fixed[y_true[idx]]}"
            )
            axes[i].axis("off")

        for j in range(len(chunk), len(axes)):
            axes[j].imshow(placeholder_array)
            axes[j].axis("off")

        save_path = os.path.join(save_dir, f"uncertain_predictions_{fig_idx}.png")
        save_or_show_figure(fig, save_path, save_instead_of_show)

# ================================================================================================================
# ============================ End Of Visualization ==============================================================
# ================================================================================================================



# ================================================================================================================
# ============================ Evaluation Helpers ================================================================
# ================================================================================================================

def find_indices_for_high_confidence_errors(y_true, y_pred, confidences, threshold, class_names_fixed, debug=False):
    """Evaluate high-confidence errors and return sorted indices."""
    high_conf_wrong = (y_pred != y_true) & (confidences >= threshold)
    error_indices = np.where(high_conf_wrong)[0]

    if debug:
        print("High confidence errors:")
        for idx in error_indices:
            print(f"Img {idx}: Pred: {class_names_fixed[y_pred[idx]]} "
                f"(Conf: {confidences[idx]:.2f}) - GT: {class_names_fixed[y_true[idx]]}")

    # sorted_indices is now an array of indices corresponding to high-confidence errors, sorted in descending order of their confidence scores.
    sorted_indices = error_indices[np.argsort(-confidences[error_indices])]
    if debug:
        print(f"Nof high confidence errors is {len(sorted_indices)}")
    return sorted_indices


def find_indices_for_uncertain_predictions(probabilities, threshold):
    """Find indices of uncertain predictions based on the difference between top-2 probabilities."""
    uncertain_indices = []
    for idx, prob in enumerate(probabilities):
        top_2_probs = np.sort(prob)[-2:]  # Get the top-2 probabilities
        if abs(top_2_probs[1] - top_2_probs[0]) < threshold:
            uncertain_indices.append(idx)
    return uncertain_indices


def compute_accuracy_topk(y_true: np.ndarray, y_pred_topk: np.ndarray, probabilities: np.ndarray = None, k: int = 1) -> float:
    """Evaluate top-k accuracy. If k is not specified it will calculate the normal accuracy."""
    # 0) Preliminary checks
    if y_true.shape[0] != y_pred_topk.shape[0]:
        raise ValueError(f"y_true and y_pred_topk must have the same number of samples. Got {y_true.shape[0]} and {y_pred_topk.shape[0]}.")
    if y_pred_topk.shape[1] != k:
        raise ValueError(f"y_pred_topk must have shape (num_samples, {k}). Got {y_pred_topk.shape}.")

    if k < 1:
        raise ValueError(f"k must be at least 1. Got {k}.")
    if k > 1 and (probabilities is None):
        raise ValueError("For k > 1, confidences and threshold must be provided to evaluate high-confidence top-k accuracy.")

    correct = 0
    for gt, pred in zip(y_true, y_pred_topk):
        # Check if the ground truth is in the top-k predictions
        if gt in pred:
            correct += 1

    return correct / len(y_true) if len(y_true) > 0 else 0.0


def compute_accuracy_keras(y_true, probabilities):
    """Evaluate accuracy using Keras metrics."""
    import tensorflow as tf

    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy_metric.update_state(y_true, probabilities)

    return accuracy_metric.result().numpy()


def compute_precision_recall_f1(y_true, y_pred, num_classes):
    """
    Compute precision, recall, F1-score, and support for each class, along with macro and weighted averages.

    Parameters
    ----------
    y_true : np.ndarray or tf.Tensor
        Ground truth labels (1D array of integers).
    y_pred : np.ndarray or tf.Tensor
        Predicted labels (1D array of integers).
    num_classes : int, optional
        Total number of classes. If None, it will be inferred from the data.

    Returns
    -------
    metrics : dict
        A dictionary containing precision, recall, F1-score, and support for each class,
        as well as macro and weighted averages.
    """
    import tensorflow as tf

    # 0) Convert to tensors if not already
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # 1) Compute metrics class by class
    class_metrics = {}
    for class_id in range(num_classes):
        # Create binary masks for the current class
        y_true_binary = tf.cast(y_true == class_id, tf.int32)
        y_pred_binary = tf.cast(y_pred == class_id, tf.int32)

        # Precision
        precision_metric = tf.keras.metrics.Precision()
        precision_metric.update_state(y_true_binary, y_pred_binary)
        precision = precision_metric.result().numpy()

        # Recall
        recall_metric = tf.keras.metrics.Recall()
        recall_metric.update_state(y_true_binary, y_pred_binary)
        recall = recall_metric.result().numpy()

        # F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Support (number of true instances for this class)
        support = tf.reduce_sum(y_true_binary).numpy()

        # Store metrics for this class
        class_metrics[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support
        }

    # 2) Compute macro averages
    macro_precision = np.mean([m["precision"] for m in class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in class_metrics.values()])
    macro_f1 = np.mean([m["f1_score"] for m in class_metrics.values()])

    # 3) Compute weighted averages
    total_support = sum([m["support"] for m in class_metrics.values() if isinstance(m, dict)])
    weighted_precision_sum = sum([m["precision"] * m["support"] for m in class_metrics.values() if isinstance(m, dict)])
    weighted_recall_sum = sum([m["recall"] * m["support"] for m in class_metrics.values() if isinstance(m, dict)])

    weighted_precision = weighted_precision_sum / total_support if total_support > 0 else 0
    weighted_recall = weighted_recall_sum / total_support if total_support > 0 else 0
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0

    # Add macro and weighted averages to the result
    class_metrics["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    }
    class_metrics["weighted_avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1
    }

    return class_metrics


def tsne_visualization():
    # TODO
    pass

# ================================================================================================================
# ============================ End Of Evaluation Helpers =========================================================
# ================================================================================================================

def evaluate_models_agreement():
    # TODO
    pass


def evaluate_keras_model(model, test_generator, model_name, save_instead_of_show=True):
    """Evaluate a Keras model and visualize high-confidence and uncertain predictions."""

    print("=======================================================================================")
    print(f"========= Evaluating {model_name} (version {ALL_MODELS_PATHS[model_name]})... =========")
    print("=======================================================================================")

    # 0) Prepare the predictions
    class_names_fixed = EMOTIONS

    y_true = np.argmax(test_generator.y_data, axis=1)
    test_generator.in_evaluate_mode = True
    probabilities = model.predict(test_generator, verbose=1)
    test_generator.in_evaluate_mode = False

    y_pred = np.argmax(probabilities, axis=1)
    y_pred_toptwo = np.argsort(probabilities, axis=1)[:, -2:]
    y_pred_topthree = np.argsort(probabilities, axis=1)[:, -3:]
    confidences = np.max(probabilities, axis=1)

    high_confindence_threshold = 0.6
    uncertain_threshold = 0.1  # Minimum difference between top-2 probabilities to consider uncertain

    # 1) Accuracy
    accuracy = compute_accuracy_topk(y_true, y_pred.reshape(-1, 1), k=1)
    accuracy_check = compute_accuracy_keras(y_true, probabilities)
    if not np.isclose(accuracy, accuracy_check, atol=1e-4):
        raise ValueError(f"Accuracy mismatch: top-1 accuracy is {accuracy:.4f} but Keras accuracy is {accuracy_check:.4f}. This indicates a potential issue in the evaluation logic.")

    accuracy_toptwo = compute_accuracy_topk(y_true, y_pred_toptwo, probabilities=probabilities, k=2)
    accuracy_topthree = compute_accuracy_topk(y_true, y_pred_topthree, probabilities=probabilities, k=3)

    accuracies = {
        "accuracy": accuracy,
        "accuracy_top2": accuracy_toptwo,
        "accuracy_top3": accuracy_topthree,
    }

    # 2) Precision, Recall, F1-score
    precision_recall_f1 = compute_precision_recall_f1(y_true, y_pred, num_classes=len(EMOTIONS))

    # 3) Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))

    # 4) High Confidence Errors
    sorted_indices = find_indices_for_high_confidence_errors(y_true, y_pred, confidences, high_confindence_threshold, class_names_fixed)

    # 5) Uncertain Predictions
    uncertain_indices = find_indices_for_uncertain_predictions(probabilities, uncertain_threshold)

    # 6) t-SNE for feature space visualization (do later as it's complicated to decide how to interpret it)
    # TODO

    # !!! Show or process results !!!
    # 1) Accuracy
    # _______________________________________________________________________________
    print("------------------------------------------------------------------------")
    print("---------- Accuracies --------------------------------------------------")
    print("------------------------------------------------------------------------")
    for key, value in accuracies.items():
        print(f"{key}: {value:.4f}")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print()

    # 2) Precision, Recall, F1-score
    # ______________________________________________________________________________
    print("------------------------------------------------------------------------")
    print("---------- Precision, Recall, F1-score (per class + macro/weighted) ----")
    print("------------------------------------------------------------------------")
    for key, value in precision_recall_f1.items():
        if isinstance(value, dict):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print()

    # 3) Confusion Matrix
    # ______________________________________________________________________________
    print("------------------------------------------------------------------------")
    print("---------- Confusion Matrix --------------------------------------------")
    print("------------------------------------------------------------------------")
    print(conf_matrix)

    # 4) High Confidence Errors
    # ______________________________________________________________________________
    print("------------------------------------------------------------------------")
    print("---------- High Confidence Errors --------------------------------------")
    print("------------------------------------------------------------------------")
    print(f"Found {len(sorted_indices)} high confidence errors with threshold >= {high_confindence_threshold}.")
    if len(sorted_indices) > 0:
        save_dir = os.path.join(SAVED_IMAGES_PATH, model_name, "high_confidence_errors", get_timestamp())
        visualize_high_confidence_errors(sorted_indices, test_generator, y_pred, y_true, confidences, class_names_fixed, save_dir, save_instead_of_show)
    else:
        print("No high confidence errors found.")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print()

    # 5) Uncertain Predictions
    # ______________________________________________________________________________
    print("------------------------------------------------------------------------")
    print("---------- Uncertain Predictions ---------------------------------------")
    print("------------------------------------------------------------------------")
    print(f"Found {len(uncertain_indices)} uncertain predictions with threshold < {uncertain_threshold}.")
    if len(uncertain_indices) > 0:
        save_dir = os.path.join(SAVED_IMAGES_PATH, model_name, "uncertain_predictions", get_timestamp())
        visualize_uncertain_predictions(uncertain_indices, test_generator, y_pred, y_true, probabilities, class_names_fixed, save_dir, save_instead_of_show)
    else:
        print("No uncertain predictions found.")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print()

    print("=======================================================================================")
    print()

    return accuracies, precision_recall_f1, probabilities, y_true, y_pred


def evaluate_yolo_model(model, test_generator):
    # Classi in formato "fisso" per la didascalia
    class_names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
    class_names_fixed = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRALITY', 'SADNESS', 'SURPRISE']

    path = [img.decode('utf-8') for img in test_generator.paths_data]

    test_folder = '/content/drive/MyDrive/Colab Notebooks/datasets/dataset_giusto/test'
    results_folder = '/content/drive/MyDrive/Colab Notebooks/HPC/finale/YOLO/finetuning'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    classes = sorted([subdir for subdir in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, subdir))])
    class_to_number = {class_name: index for index, class_name in enumerate(classes)}

    true_labels, pred_probs, image_paths = [], [], []

    # Costruiamo la lista di path e le label reali
    for img in path:
        label = img.split('_')[2]
        image_folder = os.path.join(test_folder, label, img)
        image_paths.append(image_folder)
        class_label = class_to_number[label]  # Non utilizzato direttamente
        index_label = class_names.index(label)
        true_labels.append(index_label)

    # Se necessario, ordina le true_labels
    true_labels.sort()

    print("Class to Number Mapping:", class_to_number)
    print("Classes (Alphabetical Order):", classes)
    print("True Labels (Numerical):", true_labels)

    pred_labels = []
    for category in class_names:
        results = model.predict(f'{test_folder}/{category}')
        # top1
        pred_labels.extend([result.probs.top1 for result in results])
        # Probabilità (torch.Tensor -> numpy se necessario)
        pred_probs.extend([
            result.probs.data.cpu().numpy() if isinstance(result.probs.data, torch.Tensor)
            else result.probs.data
            for result in results
        ])

    # Convertiamo in numpy array per manipolarli comodamente
    predicted_classes = np.array(pred_labels)
    true_classes = np.array(true_labels)
    probabilities = np.array(pred_probs)

    # Ricaviamo la classe predetta e la confidenza massima
    y_true = true_classes
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    # Definiamo la soglia per considerare "alta confidenza"
    threshold = 0.6
    high_conf_wrong = (y_pred != y_true) & (confidences >= threshold)
    print("Errori con alta confidenza:")
    error_indices = np.where(high_conf_wrong)[0]
    for idx in error_indices:
        print(f"Immagine {idx}: Predetto {class_names[y_pred[idx]]} "
              f"(Conf: {confidences[idx]:.2f}) - Reale: {class_names[y_true[idx]]}")

    # Ordiniamo gli errori ad alta confidenza in base alla confidenza (decrescente)
    sorted_indices = error_indices[np.argsort(-confidences[error_indices])]
    num_errors = len(sorted_indices)
    print(f"\nNumero totale di errori con confidenza >= {threshold}: {num_errors}\n")

    if num_errors > 0:
        # Dimensione del batch di subplot (3x3 = 9)
        batch_size = 18

        # Creiamo un'immagine "bianca" 224x224 per riempire subplot vuoti
        placeholder_img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        placeholder_array = np.array(placeholder_img)

        for fig_idx, start_idx in enumerate(range(0, num_errors, batch_size), start=1):
            end_idx = start_idx + batch_size
            subset_indices = sorted_indices[start_idx:end_idx]

            # Creiamo la figura 3x3, sempre uguale
            fig, axes = plt.subplots(6, 3, figsize=(9, 18))
            axes = axes.flatten()

            for i, idx in enumerate(subset_indices):
                # Carichiamo l'immagine corrispondente da image_paths
                img = Image.open(image_paths[idx]).convert('RGB')
                # Ridimensioniamo a 224x224 per avere sempre la stessa dimensione
                img_resized = img.resize((224, 224), Image.BILINEAR)
                img_array = np.array(img_resized)

                axes[i].imshow(img_array)
                axes[i].set_title(
                    f"Pred: {class_names_fixed[y_pred[idx]]}\n"
                    f"Conf: {confidences[idx]:.2f}\n"
                    f"True: {class_names_fixed[y_true[idx]]}"
                )
                axes[i].axis("off")

            # Riempie i subplot vuoti con placeholder bianco
            for j in range(len(subset_indices), 9):
                axes[j].imshow(placeholder_array)
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(f'{results_folder}/bias_{fig_idx}.png')
            plt.show()
    else:
        print("Nessun errore con alta confidenza trovato.")

    return probabilities, y_true, y_pred, image_paths

