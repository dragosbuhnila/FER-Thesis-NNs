import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from modules.config import EMOTIONS

def evaluate_keras_model(model, test_generator, model_name):
    # class_names_fixed = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRALITY', 'SADNESS', 'SURPRISE']
    class_names_fixed = EMOTIONS

    y_true = np.argmax(test_generator.y_data, axis=1)
    test_generator.in_evaluate_mode = True
    probabilities = model.predict(test_generator, verbose=1)
    test_generator.in_evaluate_mode = False

    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    threshold = 0.6
    high_conf_wrong = (y_pred != y_true) & (confidences >= threshold)

    print("Errori con alta confidenza:")
    error_indices = np.where(high_conf_wrong)[0]
    for idx in error_indices:
        print(f"Immagine {idx}: Pred: {class_names_fixed[y_pred[idx]]} "
              f"(Conf: {confidences[idx]:.2f}) - Reale: {class_names_fixed[y_true[idx]]}")

    sorted_indices = error_indices[np.argsort(-confidences[error_indices])]
    num_errors = len(sorted_indices)
    print(f"Numero totale di errori ad alta confidenza: {num_errors}")

    if num_errors > 0:
        batch_size = 18  # 6 righe x 3 colonne

        # Prepara un'immagine "bianca" 224x224 da usare come placeholder
        placeholder_img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        placeholder_array = np.array(placeholder_img)

        for fig_idx, start in enumerate(range(0, num_errors, batch_size), start=1):
            end = start + batch_size
            chunk = sorted_indices[start:end]

            # Creiamo la figura sempre della stessa dimensione
            fig, axes = plt.subplots(6, 3, figsize=(9, 18))
            axes = axes.flatten()

            for i, idx in enumerate(chunk):
                # Carichiamo l'immagine dal generatore
                img_data = test_generator.x_data[idx].astype('uint8')

                # Ridimensioniamo a 224x224
                pil_img = Image.fromarray(img_data).resize((224, 224), Image.BILINEAR)
                resized_img = np.array(pil_img)

                axes[i].imshow(resized_img)
                axes[i].set_title(
                    f"Pred: {class_names_fixed[y_pred[idx]]}\n"
                    f"Conf: {confidences[idx]:.2f}\n"
                    f"True: {class_names_fixed[y_true[idx]]}"
                )
                axes[i].axis("off")

            # Riempie i subplot rimanenti con l'immagine bianca
            for j in range(len(chunk), 18):
                axes[j].imshow(placeholder_array)
                axes[j].axis("off")

            # Se noti comportamenti indesiderati, puoi commentare tight_layout
            plt.tight_layout()

            plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/HPC/finale/{model_name}/finetuning/bias_{fig_idx}.png')
            plt.show()
    else:
        print("Nessun errore con alta confidenza trovato.")

    return probabilities, y_true, y_pred


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

