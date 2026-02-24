import os
import random
import time
import numpy as np
from PIL import Image
import scipy
from tqdm import tqdm
import shutil
import zipfile
from joblib import Parallel, delayed

from modules.config import EMOTIONS
from modules.misc import get_timestamp, zip_folder 
from modules.model import load_model



def create_gaussian_bubbles_mask(image_array, bubble_radius, num_bubbles):
    """
    Crea una maschera con "bolle" gaussiane (bubble_radius e num_bubbles).
    Le aree in cui l'immagine è nera ( [0,0,0] ) sono escluse (non feasible).
    """
    mask = np.zeros(image_array.shape[:2], dtype=np.float32)
    height, width = mask.shape

    # Sigma proporzionale al raggio
    fraction = random.uniform(3, 3.1)
    sigma = bubble_radius / fraction
    bubble_centers = []

    # Points initialized as feasible but then updated to unfeasible if they are black or within existing bubbles
    feasible_points = np.ones((height, width), dtype=bool)
    feasible_points[np.all(image_array == [0, 0, 0], axis=-1)] = False
    for _ in range(num_bubbles):
        for cx, cy in bubble_centers:
            y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            distance_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            feasible_points[distance_from_center < bubble_radius] = False

        feasible_indices = np.argwhere(feasible_points)
        if len(feasible_indices) == 0:
            # Se non ci sono più punti feasible, finisci
            break

        # Scegli un punto feasible a caso come centro bolla
        y, x = feasible_indices[np.random.choice(len(feasible_indices))]
        bubble_centers.append((x, y))

        # Costruisci la gaussiana intorno al centro
        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))

        # Azzera la gaussiana oltre il raggio
        distance_from_center = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        gaussian[distance_from_center > bubble_radius] = 0

        # Final mask will be justu the addition of the current gaussian. Note that gaussian values are between 0 and 1 since it's a probability distribution.
        mask = np.maximum(mask, gaussian)

    return mask


def get_masks(image_array, bubble_radius, num_bubbles, iterations):
    """
    Given an image, it creates various masked versions of it, each containing the same number and same radius for bubbles.
    Ritorna:
      masked_images: array (iterations x H x W x 3) con le immagini mascherate
      masks: array (iterations x H x W) con le maschere
    """
    height, width, rgb = image_array.shape
    masked_images = np.zeros((iterations, height, width, rgb))
    masks = np.zeros((iterations, height, width))

    for i in range(iterations):
        mask = create_gaussian_bubbles_mask(image_array, bubble_radius, num_bubbles)
        masks[i] = mask

        # Applico la maschera
        mask_rgb = np.stack([mask] * 3, axis=-1)
        masked_image = image_array / 255.0 * mask_rgb

        masked_images[i] = masked_image
            
    return masked_images, masks


def get_batch_planes(masked_images, masks, model, labels):
    """
    Esegue la predizione su tutte le immagini mascherate e,
    per ogni immagine e per ogni classe c, salva (mask * p_c) in batch_planes[c].

    Ritorna:
    - batch_planes: lista di liste, batch_planes[c] = [lista di mask*prob per la classe c]
    - mask_classes: array con le argmax classi per ogni immagine mascherata
    """
    # Prepara le immagini per il modello
    masked_steps = masked_images * 255.0
    mask_preds = model.predict(masked_steps, verbose=0) # shape: (iterations, num_classes)

    # Argmax per valutare "quante volte" la classe originale si conserva
    mask_classes = np.argmax(mask_preds, axis=-1) # takes the argmax along the class dimension, resulting in shape (iterations,)

    # Inizializzo i piani per ciascuna classe
    batch_planes = [[] for _ in range(len(labels))]

    # Per ogni immagine mascherata, aggiungo la maschera pesata per TUTTE le classi
    for i in range(len(mask_classes)):
        for c in range(len(labels)):
            prob_c = mask_preds[i, c]
            weighted_mask = masks[i] * prob_c
            batch_planes[c].append(weighted_mask) # batch_planes[c] è una lista di maschere pesate per la classe c

    return batch_planes, mask_classes


def adjust_probabilities(probabilities, num_bubbles, bubble_range, accuracy_diff, good):
    """
    Aggiusta la distribuzione con cui scelgo il numero di bubble
    in base a quanto l'accuracy si discosta dal target.
    """
    index = bubble_range.index(num_bubbles)

    # Fattore di aggiustamento
    if good:
        adjustment_factor = 0.5 - accuracy_diff
    else:
        adjustment_factor = -2 * accuracy_diff

    # Distribuzione gaussiana centrata su 'index'
    std_dev = 1.5
    gaussian = scipy.stats.norm(loc=index, scale=std_dev)

    # Aggiorno le probabilità
    for i in range(len(probabilities)):
        influence = gaussian.pdf(i)
        probabilities[i] += adjustment_factor * influence

    # Clipping e normalizzazione
    probabilities = [max(0.01, p) for p in probabilities]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return probabilities


def calculate_average_plane(planes):
    """
    Esegue una semplice media delle maschere in planes.
    ATTENZIONE: se i planes sono già weighted (mask * p_c),
    questa media non è esattamente la “expectation”.
    Per l’uso dimostrativo va bene.
    """
    if len(planes) > 0:
        return np.mean(planes, axis=0)
    return None


def normalize_image(image):
    """ Normalizza un'immagine fra 0 e 1 """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    else:
        return image


def process_image(image_array, model, class_names_fixed, bubble_radius, iterations, accuracy_target, accuracy_tolerance, predicted_index, predicted_probability):
    """
    Process a single image to generate bubble masks and planes.
    Returns the predicted class, probability, and generated planes.
    """
    # check that image_array is a single image and not a batch
    if len(image_array.shape) != 3:
        raise ValueError(f"Expected a single image array with shape (H, W, C), but got a batch with shape {image_array.shape}.")

    # Use the precomputed predicted_index and predicted_probability from the test generator predictions instead of recomputing them here. This ensures consistency and avoids redundant model predictions.
    # print(f"\tPredicted class: {class_names_fixed[predicted_index]} with probability: {predicted_probability*100:.2f}%")

    all_planes = [[] for _ in range(len(class_names_fixed))]
    lengths = np.zeros(len(class_names_fixed)).astype(int)

    bubble_range = list(range(5, 26))
    probabilities = [1 / len(bubble_range)] * len(bubble_range)
    history_of_num_bubbles = []

    while True:
        num_bubbles = np.random.choice(bubble_range, p=probabilities)
        history_of_num_bubbles.append(num_bubbles)
        # print(f"\tTesting with {num_bubbles} bubbles...")

        masked_images, masks = get_masks(image_array, bubble_radius, num_bubbles, iterations)
        batch_planes, mask_classes = get_batch_planes(masked_images, masks, model, class_names_fixed)

        true_plane_len = sum(1 for c in mask_classes if c == predicted_index)
        iteration_accuracy = true_plane_len / iterations
        # print(f"\tIteration accuracy: {iteration_accuracy:.2f}")

        good = accuracy_tolerance <= iteration_accuracy <= (1 - accuracy_tolerance)
        accuracy_diff = abs(iteration_accuracy - accuracy_target)
        probabilities = adjust_probabilities(probabilities, num_bubbles, bubble_range, accuracy_diff, good)

        for p, batch_plane in enumerate(batch_planes):
            all_planes[p].extend(batch_plane)
            lengths[p] += len(batch_plane)

        if np.sum(lengths) > 4000:
            break

    return all_planes, history_of_num_bubbles


def save_planes_and_images(image_name, image_array, predicted_index, predicted_probability, all_planes, labels, output_subfolder, history_of_num_bubbles):
    """
    Save the generated planes and images to the output folder.
    """
    false_planes = []
    for i, planes_list in enumerate(all_planes):
        avg_plane = calculate_average_plane(planes_list)
        if avg_plane is not None:
            normalized_plane = normalize_image(avg_plane)
            norm_plane_img = Image.fromarray((normalized_plane * 255).astype(np.uint8))
            norm_plane_img.save(os.path.join(
                output_subfolder,
                f"{image_name}_normplane_{labels[i]}_numerosity_{len(planes_list)}.png"
            ))
        if i != predicted_index and avg_plane is not None:
            false_planes.append(avg_plane)

    if len(false_planes) > 0:
        global_false = np.mean(false_planes, axis=0)
        normalized_false_plane = normalize_image(global_false)
        norm_false_img = Image.fromarray((normalized_false_plane * 255).astype(np.uint8))
        norm_false_img.save(os.path.join(
            output_subfolder,
            f"{image_name}_normglobalfalseplane.png"
        ))

    original_image = Image.fromarray(image_array)
    original_image.save(os.path.join(
        output_subfolder,
        f"{image_name}__predictedclass_{labels[predicted_index]}_"
        f"withprob_{predicted_probability*100:.2f}_"
        f"avgbubblesnum_{np.mean(history_of_num_bubbles):.2f}.png"
    ))


def process_single_image(idx, image_array, label, predicted_index, predicted_probability, model_name, class_names_fixed, bubble_radius, iterations, accuracy_target, accuracy_tolerance, output_folder):
    """
    Process a single image and save the results.
    This function is designed to be used in parallel processing.
    """
    model = load_model(model_name)

    image_name = f"image_{idx}"
    output_subfolder = os.path.join(output_folder, class_names_fixed[label])
    os.makedirs(output_subfolder, exist_ok=True)

    all_planes, history_of_num_bubbles = process_image(
        image_array, model, class_names_fixed, bubble_radius, iterations, accuracy_target, accuracy_tolerance, predicted_index, predicted_probability
    )
    save_planes_and_images(
        image_name, image_array, predicted_index, predicted_probability, all_planes, class_names_fixed, output_subfolder, history_of_num_bubbles
    )


def generate_bubbles_planes(model: object, model_name: str, test_generator: object, 
                            output_base_folder_path: str, run_name: str = None,
                            iterations: int = 200, bubble_radius: int = 26, accuracy_target: float = 0.5, accuracy_tolerance: float = 0.3,
                            n_jobs=4):
    """
        Generate bubble-based explanations and plane visualizations for model predictions on test images.
        This function processes a test dataset through a model, generating bubble-based visual explanations
        and corresponding plane data for each image. Results are organized by emotion labels and saved to disk.
        Args:
            model (object): Trained model object with prediction capabilities.
            model_name (str): Name of the model, used for output folder organization.
            test_generator (object): Generator object that yields batches of (image, label) tuples from test data.
            output_base_folder_path (str): Base path for saving output results.
            iterations (int, optional): Number of iterations for bubble generation algorithm. Defaults to 200.
            bubble_radius (int, optional): Radius of bubbles in pixels. Defaults to 26.
            accuracy_target (float, optional): Target accuracy threshold for bubble placement. Defaults to 0.5.
            accuracy_tolerance (float, optional): Tolerance range for accuracy threshold. Defaults to 0.3.
        Returns:
            None: Saves results to disk and creates a zip archive.
        Side Effects:
            - Creates subdirectories in output_folder organized by emotion labels.
            - Saves plane images and bubble data for each test image.
            - Creates a zip archive of all generated results.
            - Prints progress updates with estimated remaining processing time.
        Raises:
            None explicitly, but may raise exceptions from process_image, save_planes_and_images, or file I/O operations.
    """
    class_names_fixed = EMOTIONS

    run_name = run_name if run_name else f"{get_timestamp()}_bubbles"

    # e.g. results_light/saved_images/20260220-164405_bubbles/occft_convnext
    run_folder = os.path.join(output_base_folder_path, run_name)
    output_folder = os.path.join(run_folder, model_name) 
    os.makedirs(output_folder, exist_ok=True)

    images = test_generator.x_data

    gt_probabilities = test_generator.y_data
    labels = np.argmax(gt_probabilities, axis=1)

    predicted_probabilities = model.predict(test_generator)
    predicted_indices = np.argmax(predicted_probabilities, axis=1)
    predicted_probabilities = np.max(predicted_probabilities, axis=1)

    print(f"[INFO] Processing {len(labels)} images with model '{model_name}'...")
    # The structure will be:
    # output_folder_for_the_run/    (format is timestamp_bubbles_settings...)
    #    model_name/                (e.g. occft_convnext)
    #      emotion_gt/              (e.g. HAPPY)
    #        images_names_with_three_formats
    # Parallelize the processing of images
    Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(
            idx, image_array, label, predicted_index, predicted_probability, model_name, class_names_fixed, bubble_radius, iterations, accuracy_target, accuracy_tolerance, output_folder
        )
        for idx, (image_array, label, predicted_index, predicted_probability) in tqdm(
            enumerate(zip(images, labels, predicted_indices, predicted_probabilities)),
            total=len(labels),
            desc="Processing images",
            unit="image"
        )
    )
    zip_folder(output_folder, os.path.join(run_folder, f"{model_name}_bubbles.zip"))