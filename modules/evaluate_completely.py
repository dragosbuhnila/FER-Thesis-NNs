import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from modules.config import EMOTIONS, SAVED_IMAGES_PATH, ALL_MODELS_PATHS
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


def visualize_confusion_matrix(cm, class_names, model_name, save_dir, save_instead_of_show):
    """
    Visualize and save the confusion matrix and its normalized version.

    Parameters:
    - cm: Confusion matrix (2D array).
    - class_names: List of class names for the axes.
    - model_name: Name of the model (used for saving the plots).
    - save_dir: Directory where the plots will be saved.
    - save_instead_of_show: Whether to save the figure instead of showing it.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Compute normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot and save the confusion matrix
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join(save_dir, f'{model_name}_cm.png')
    save_or_show_figure(fig, cm_path, save_instead_of_show)

    # Plot and save the normalized confusion matrix
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_normalized_path = os.path.join(save_dir, f'{model_name}_cm_normalized.png')
    save_or_show_figure(fig, cm_normalized_path, save_instead_of_show)


def visualize_tsne_feature_space(reduced_features, labels, class_names, model_name, save_dir, save_instead_of_show, perplexity=30):
    os.makedirs(save_dir, exist_ok=True)

    # Map numeric labels to class names
    class_labels = [class_names[label] for label in labels]

    # Plot t-SNE with distinct colors
    fig = plt.figure(figsize=(10, 8))  # Create a Figure object
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=class_labels, palette='Set1', s=50, edgecolor='k')
    plt.title(f't-SNE Visualization of Features with perplexity {perplexity}')
    plt.legend(title='Class', loc='best')

    # Save the plot
    save_path = os.path.join(save_dir, f"tsne_{model_name}_perplexity_{perplexity}.png")
    save_or_show_figure(fig, save_path, save_instead_of_show)  # Pass the Figure object

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


def compute_precision_recall_f1(y_true, y_pred, num_classes, fixed_class_names):
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
        class_name = fixed_class_names[class_id] 
        class_metrics[class_name] = {
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


def compute_tsne_reduced_features(features, perplexity=30):
    """
    Visualize features reduced with t-SNE and display class names in the legend with distinct colors.
    Although we call them features, they are actually the model's output probabilities for each class, meaning
    we're mostly interested in visualizing how the model's confidence distribution looks in a 2D space, and whether 
    samples from different classes cluster together based on their predicted probabilities.

    Args:
    - features: The features to be reduced with t-SNE.
    """
    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    tsne_reduced_features = tsne.fit_transform(features)

    return tsne_reduced_features

# ================================================================================================================
# ============================ End Of Evaluation Helpers =========================================================
# ================================================================================================================



# ================================================================================================================
# ========================= Complete Model Evaluation Functions ==================================================
# ================================================================================================================


def evaluate_keras_model(model, test_generator, model_name, save_instead_of_show=True, run_name=None):
    """Evaluate a Keras model and visualize high-confidence and uncertain predictions."""

    print("=======================================================================================")
    print(f"========= Evaluating {model_name} ========")
    print("=======================================================================================")
    run_name = get_timestamp() if run_name is None else run_name
    base_dir = os.path.join(SAVED_IMAGES_PATH, run_name, model_name)

    print(f"[INFO] time reference: {run_name}")
    print(f"[INFO] model_version: {ALL_MODELS_PATHS[model_name]}")

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
    uncertain_threshold = 0.1               # Minimum difference between top-2 probabilities to consider uncertain

    # 1) Accuracy
    print("[INFO] Evaluating accuracies...")
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
    print("[INFO] Evaluating precision, recall, and F1-score...")
    precision_recall_f1 = compute_precision_recall_f1(y_true, y_pred, num_classes=len(EMOTIONS), fixed_class_names=class_names_fixed)

    # 3) Confusion Matrix
    print("[INFO] Computing confusion matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))

    # 4) High Confidence Errors
    print("[INFO] Identifying high-confidence errors...")
    sorted_indices = find_indices_for_high_confidence_errors(y_true, y_pred, confidences, high_confindence_threshold, class_names_fixed)

    # 5) Uncertain Predictions
    print("[INFO] Identifying uncertain predictions...")
    uncertain_indices = find_indices_for_uncertain_predictions(probabilities, uncertain_threshold)

    # 6) t-SNE for feature space visualization (do later as it's complicated to decide how to interpret it)
    perplexity_values = [5, 20, 30, 50]
    print(f"[INFO] Computing t-SNE reduced features with perplexities {perplexity_values}...")
    tsne_results = {}
    for perplexity in perplexity_values:
        tsne_reduced_features = compute_tsne_reduced_features(features=probabilities, perplexity=perplexity)
        tsne_results[perplexity] = tsne_reduced_features


    print()
    # !!! Show or process results !!!
    # 1) Accuracy
    # _______________________________________________________________________________
    print("-" * 50)
    print("1) Accuracies:")
    for key, value in accuracies.items():
        print(f"\t{key}: {value:.4f}")
    print()

    # Save accuracies to CSV
    accuracies_csv_path = os.path.join(base_dir, f"accuracies.csv")
    os.makedirs(os.path.dirname(accuracies_csv_path), exist_ok=True)
    pd.DataFrame(accuracies.items(), columns=["Metric", "Value"]).to_csv(accuracies_csv_path, index=False)
    print(f"[INFO] Saved accuracies to CSV: {accuracies_csv_path}")
    print()

    # 2) Precision, Recall, F1-score
    # _______________________________________________________________________________
    print("-" * 50)
    print("2) Precision, Recall, F1-score (per class + macro/weighted):")
    precision_recall_f1_data = []
    for key, value in precision_recall_f1.items():
        if isinstance(value, dict):
            print(f"\t{key}")
            for sub_key, sub_value in value.items():
                print(f"\t\t{sub_key}: {sub_value:.4f}")
                precision_recall_f1_data.append({"Class": key, "Metric": sub_key, "Value": sub_value})
        else:
            raise ValueError(f"Expected precision_recall_f1 to be a dict of dicts, but got {type(precision_recall_f1)}")
    print()

    # Save precision-recall-f1 to CSV
    precision_recall_f1_csv_path = os.path.join(base_dir, f"precision_recall_f1.csv")
    pd.DataFrame(precision_recall_f1_data).to_csv(precision_recall_f1_csv_path, index=False)
    print(f"[INFO] Saved precision-recall-f1 to CSV: {precision_recall_f1_csv_path}")
    print()

    # 3) Confusion Matrix
    # _______________________________________________________________________________
    print("-" * 50)
    print("3) Confusion Matrix:")
    confusion_matrix_save_dir = os.path.join(base_dir, "confusion_matrix")
    visualize_confusion_matrix(conf_matrix, class_names_fixed, model_name, confusion_matrix_save_dir, save_instead_of_show)
    print(f"[INFO] Confusion matrix visualizations saved to: {confusion_matrix_save_dir}")
    print()

    # 4) High Confidence Errors
    # _______________________________________________________________________________
    print("-" * 50)
    print("4) High Confidence Errors:")
    if len(sorted_indices) > 0:
        print(f"[INFO] Found {len(sorted_indices)} high confidence errors with threshold >= {high_confindence_threshold}.")
        high_confidence_errors_save_dir = os.path.join(base_dir, "high_confidence_errors")
        visualize_high_confidence_errors(sorted_indices, test_generator, y_pred, y_true, confidences, class_names_fixed, high_confidence_errors_save_dir, save_instead_of_show)
        print(f"[INFO] High confidence errors visualizations saved to: {high_confidence_errors_save_dir}")
    else:
        print("[INFO] No high confidence errors found.")
    print()

    # 5) Uncertain Predictions
    # _______________________________________________________________________________
    print("-" * 50)
    print("5) Uncertain Predictions:")
    if len(uncertain_indices) > 0:
        print(f"[INFO] Found {len(uncertain_indices)} uncertain predictions with difference threshold < {uncertain_threshold}.")
        uncertain_predictions_save_dir = os.path.join(base_dir, "uncertain_predictions")
        visualize_uncertain_predictions(uncertain_indices, test_generator, y_pred, y_true, probabilities, class_names_fixed, uncertain_predictions_save_dir, save_instead_of_show)
        print(f"[INFO] Uncertain predictions visualizations saved to: {uncertain_predictions_save_dir}")
    else:
        print("[INFO] No uncertain predictions found.")
    print()

    # 6) t-SNE Visualization
    # _______________________________________________________________________________
    print("-" * 50)
    print("6) t-SNE Visualization of feature space:")
    tsne_save_dir = os.path.join(base_dir, "tsne")
    for perplexity, tsne_reduced_features in tsne_results.items():
        visualize_tsne_feature_space(tsne_reduced_features, y_true, class_names_fixed, model_name, tsne_save_dir, save_instead_of_show, perplexity=perplexity)
        print(f"[INFO] t-SNE visualizations saved to: {tsne_save_dir}")
        print()

    # 7) Save probabilities, y_true, y_pred for potential further analysis
    # _______________________________________________________________________________
    print("-" * 50)
    print("7) Saving probabilities, y_true, and y_pred to CSV for further analysis...")
    probs_csv_path = os.path.join(base_dir, f"probs_ytrue_ypred.csv")

    # Create a DataFrame with probabilities for all classes
    probs_data = pd.DataFrame(probabilities, columns=[f"Prob_{class_name}" for class_name in class_names_fixed])
    probs_data["True_Label"] = [class_names_fixed[label] for label in y_true]
    probs_data["Predicted_Label"] = [class_names_fixed[label] for label in y_pred]
    probs_data["Prediction_Confidence"] = confidences

    # Save the DataFrame to CSV
    os.makedirs(os.path.dirname(probs_csv_path), exist_ok=True)
    probs_data.to_csv(probs_csv_path, index=False)
    print(f"[INFO] Saved probabilities, y_true, and y_pred to CSV: {probs_csv_path}")
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

# ================================================================================================================
# ========================= End Of Complete Model Evaluation Functions ===========================================
# ================================================================================================================


# ================================================================================================================
# ========================= Agreement Analysis Functions =========================================================
# ================================================================================================================

def collect_predictions_from_models(model_and_names, test_generator):    
    class_names_fixed = EMOTIONS
    true_classes = np.argmax(test_generator.y_data, axis=1)

    all_predictions = []    
    for model_name, model in model_and_names.items():
        print(f"Running predictions for: {model_name}")
        
        probs = model.predict(test_generator, verbose=1)
        preds = np.argmax(probs, axis=1)
        
        df_model = pd.DataFrame({
            "Image_Idx": range(len(true_classes)),
            "True_Label": [class_names_fixed[i] for i in true_classes],
            "Predicted_Label": [class_names_fixed[i] for i in preds],
            "Model_Name": model_name
        })
        
        all_predictions.append(df_model)
    
    df_all = pd.concat(all_predictions, ignore_index=True)
    
    return df_all


def compute_agreement_all_images(df_all):
    """
    Compute agreement P_i for each image.
    
    Returns:
        df_agreement (one row per image)
    """
    results = []
    
    grouped = df_all.groupby("Image_Idx")
    
    for image_idx, group in grouped:
        
        true_label = group["True_Label"].iloc[0]
        preds = group["Predicted_Label"].values
        
        n = len(preds)
        
        if n < 2:
            overall_pi = 0.0
            pi_dict = {emo: 0.0 for emo in EMOTIONS}
        else:
            counts = group["Predicted_Label"].value_counts()
            
            numerator = 0
            pi_dict = {}
            
            for emo in EMOTIONS:
                c = counts.get(emo, 0)
                value = c*(c-1) / (n*(n-1))
                pi_dict[emo] = value
                numerator += c*(c-1)
            
            overall_pi = numerator / (n*(n-1))
        
        results.append({
            "Image_Idx": image_idx,
            "True_Label": true_label,
            "Pi_Overall": overall_pi,
            **{f"Pi_{emo}": pi_dict[emo] for emo in EMOTIONS}
        })
    
    return pd.DataFrame(results)


def agreement_statistics(df_agreement):    
    strong = df_agreement[df_agreement["Pi_Overall"] > 0.9]
    good = df_agreement[(df_agreement["Pi_Overall"] > 0.5) & 
                          (df_agreement["Pi_Overall"] <= 0.9)]
    poor = df_agreement[df_agreement["Pi_Overall"] <= 0.3]
    
    return strong, good, poor


def plot_disagreement_images(df_all, df_agreement, test_generator, save_dir, save_instead_of_show, images_per_fig=18):
    os.makedirs(save_dir, exist_ok=True)
    
    df_plot = df_agreement.sort_values("Pi_Overall", ascending=False)
    
    plot_data = []
    
    for _, row in df_plot.iterrows():
        
        image_idx = row["Image_Idx"]
        true_label = row["True_Label"]
        pi_total = row["Pi_Overall"]
        
        group = df_all[df_all["Image_Idx"] == image_idx]
        wrong = group[group["Predicted_Label"] != group["True_Label"]]
        total_wrong = len(wrong)
        
        if total_wrong <= 1:
            continue
        
        counts_wrong = wrong["Predicted_Label"].value_counts()
        most_frequent = counts_wrong.idxmax() if len(counts_wrong)>0 else None
        
        details_str = ""
        for emo in EMOTIONS:
            details_str += f"{emo}: {counts_wrong.get(emo,0)}\n"
        
        # find emotion with max Pi
        pi_cols = [f"Pi_{emo}" for emo in EMOTIONS]
        emotion_max = row[pi_cols].idxmax().replace("Pi_","")
        pi_max = row[f"Pi_{emotion_max}"]
        
        plot_data.append({
            "image_idx": image_idx,
            "true_label": true_label,
            "pi_total": pi_total,
            "pi_max": pi_max,
            "emotion_max": emotion_max,
            "total_wrong": total_wrong,
            "most_frequent": most_frequent,
            "details_str": details_str
        })
    
    num_chunks = math.ceil(len(plot_data)/images_per_fig)
    
    for chunk_idx in range(num_chunks):
        chunk = plot_data[chunk_idx*images_per_fig:(chunk_idx+1)*images_per_fig]
        
        fig, axes = plt.subplots(6,3, figsize=(12,30))
        axes = axes.flatten()
        
        for i, data_item in enumerate(chunk):
            ax = axes[i]
            
            image_index = data_item["image_idx"]
            
            img_data = test_generator.x_data[image_index].astype('uint8')
            pil_img = Image.fromarray(img_data).resize((224, 224), Image.BILINEAR)
            img = np.array(pil_img)

            ax.imshow(img)
            ax.axis("off")
            
            ax.set_title(
                f"Misclassified by {data_item['total_wrong']} models\n"
                f"Overall Agreement: {data_item['pi_total']*100:.2f}%\n"
                f"Agreement w/ {data_item['emotion_max']}: {data_item['pi_max']*100:.2f}%\n"
                f"Most common wrong: {data_item['most_frequent']}\n"
                f"Correct: {data_item['true_label']}\n\n"
                f"{data_item['details_str']}",
                fontsize=9
            )
        
        for j in range(len(chunk), images_per_fig):
            axes[j].axis("off")
        
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"disagreement_chunk_{chunk_idx + 1}.png")
        save_or_show_figure(fig, save_path, save_instead_of_show)


def evaluate_agreement(model_and_names, test_generator, run_name=None):
    run_name = get_timestamp() if run_name is None else run_name
    base_dir = os.path.join(SAVED_IMAGES_PATH, run_name, "agreement_analysis")

    df_all = collect_predictions_from_models(model_and_names, test_generator)
    df_agreement = compute_agreement_all_images(df_all)

    # 1) save to csv the agreement values for potential further analysis
    agreement_csv_path = os.path.join(base_dir, f"agreement_values.csv")
    os.makedirs(os.path.dirname(agreement_csv_path), exist_ok=True)
    df_agreement.to_csv(agreement_csv_path, index=False)
    print(f"[INFO] Saved agreement values to CSV: {agreement_csv_path}")

    # 2) print and save agreement statistics
    strong, good, poor = agreement_statistics(df_agreement)
    total = len(df_agreement)
    print("===== AGREEMENT STATISTICS =====")
    print(f"Strong (>0.9): {len(strong)} ({len(strong)/total*100:.2f}%)")
    print(f"Good (0.5–0.9): {len(good)} ({len(good)/total*100:.2f}%)")
    print(f"Poor (<=0.3): {len(poor)} ({len(poor)/total*100:.2f}%)")
    agreement_group_stats = pd.DataFrame({
        "Agreement_Level": ["Strong", "Good", "Poor"],
        "Count": [len(strong), len(good), len(poor)],
        "Percentage": [len(strong)/total*100, len(good)/total*100, len(poor)/total*100]
    })
    agreement_stats_csv_path = os.path.join(base_dir, f"agreement_statistics.csv")
    agreement_group_stats.to_csv(agreement_stats_csv_path, index=False)
    print(f"[INFO] Saved agreement statistics to CSV: {agreement_stats_csv_path}")

    # 3) plot disagreement images
    print("\nPlotting disagreement images...")
    disagreement_save_dir = os.path.join(base_dir, "disagreeing_images")
    plot_disagreement_images(df_all, df_agreement, test_generator, disagreement_save_dir, save_instead_of_show=True)

    return df_all, df_agreement