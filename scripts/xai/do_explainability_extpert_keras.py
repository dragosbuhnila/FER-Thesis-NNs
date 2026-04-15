import os; import sys

from matplotlib import category;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modules.model import load_model
from modules.data__load import load_test_generator
from modules.misc import get_timestamp, Tee, hash_image, make_xai_result_image_name
from modules.misc import TEST_SET_CHOICES, TEST_SET_PATHS
from modules.config import (
    ALL_MODELS_PATHS,
    CONSOLE_OUTPUTS_PATH,
    SAVED_IMAGES_PATH,
    EMOTIONS,
)




def create_black_square_images(img_array, square_sizes):
    """
    Creates images with a black square placed only if the center of the square is over a non-zero pixel.
    Skips generating images in the 8-connected pixels around any center where a square has already been generated.

    :param image_path: Path to the input image.
    :param output_folder: Folder to save the output images.
    :param square_size: Size of the black square to be placed on the image.
    """

    for square_size in square_sizes:
        # Ensure the square size is odd
        if square_size % 2 == 0:
            square_size += 1  # Make it odd by adding 1

        neighbors = square_size // 5  # Number of neighbors to check

        height, width, channels = img_array.shape

        # Offset for the center of the square
        offset = square_size // 2

        # Create a boolean mask to track visited centers
        visited = np.zeros((height, width), dtype=bool)

        total_positions = (height - square_size + 1) * (width - square_size + 1)

        masked_images = []

        with tqdm(total=total_positions, desc=f"Generating images with black squares of size {square_size}") as pbar:
            counter = 0
            for y in range(offset, height - offset):
                for x in range(offset, width - offset):
                    # Skip this center if it or any of its 8-connected neighbors have already been visited
                    if visited[y, x]:
                        pbar.update(1)
                        continue

                    # Check if the center pixel of the current square is non-zero
                    if np.any(img_array[y, x] != 0):
                        # Create a copy of the original image
                        masked_image = img_array.copy()

                        # Apply the black square by setting pixels in the square to 0
                        masked_image[y-offset:y+offset+1, x-offset:x+offset+1] = 0  # Black square (RGB [0, 0, 0])

                        # Convert the image array to uint8 if it's not already
                        masked_image = masked_image.astype(np.uint8)

                        # Convert the NumPy array to a PIL Image object
                        masked_image = Image.fromarray(masked_image)

                        # Save the image as an RGB PNG file
                        masked_images.append(np.array(masked_image))
                        counter += 1

                        # Mark the center and its neighbors as visited ## immensa riduzione dei costi computazionali
                        for i in range(-neighbors, neighbors+1):  # -1, 0, 1 if n = 1
                            for j in range(-neighbors, neighbors+1):  # -1, 0, 1 if n = 1
                                if 0 <= y + i < height and 0 <= x + j < width:
                                    visited[y + i, x + j] = True

                    pbar.update(1)
    return masked_images

def calculate_saliency_map(model, model_name, original_image, perturbed_images, image_probability, class_index):
    """
    Calculate the saliency map based on the difference in predictions.
    """
    saliency_map = np.zeros(original_image.shape[:2], dtype=np.float32)

    # Predict all perturbed images as a batch
    if "yolo" in model_name.lower():
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = model.predict(list(perturbed_images), device=device)
        perturbed_predictions = [result.probs.data.cpu().numpy() for result in results]
    else:
        perturbed_predictions = model.predict(np.array(perturbed_images))

    # Calculate the differences for each perturbed image
    for perturbed_image, perturbed_prediction in zip(perturbed_images, perturbed_predictions):
        perturbed_prob = perturbed_prediction[class_index]
        difference = image_probability - perturbed_prob

        mask = (original_image != perturbed_image).any(axis=-1)
        saliency_map[mask] += difference

    return saliency_map / len(perturbed_images)

def blur_saliency_map(saliency_map, sigma):
    """
    Apply Gaussian blur to the saliency map.
    """
    return scipy.ndimage.gaussian_filter(saliency_map, sigma=sigma)

def save_saliency_map(saliency_map, output_path):
    """
    Save the saliency map as a NumPy file and an image.
    """
    np.save(output_path + ".npy", saliency_map)
    normalized_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    Image.fromarray((normalized_map * 255).astype(np.uint8)).save(output_path + ".png")



# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Generate saliency maps using external perturbations.")
parser.add_argument('--quick', action='store_true', help="Run on a small subset of the test data (1 batch).")
parser.add_argument('--redirect_output', action='store_true', help="Redirect console output to a log file.")
parser.add_argument('--models_set', type=str, choices=['occft', 'federica', 'yolo_fede', 'occft_yolo'], help="Specify which set of models to use.")
parser.add_argument('--test_set', type=str, choices=TEST_SET_CHOICES, help="Specify which test set to use.")
parser.add_argument('--output_folder', type=str, help="Base folder path for saving saliency maps.")
parser.add_argument('--square_sizes', type=int, nargs='+', default=[35, 27, 19], help="Sizes of black squares for perturbations.")
parser.add_argument('--blur_sigma', type=float, default=2.0, help="Sigma value for Gaussian blurring.")
args = parser.parse_args()

# MODELS
if args.models_set == 'occft':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
elif args.models_set == 'federica':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "finetuning" in model_name.lower()]
elif args.models_set == 'yolo_fede':
    MODEL_NAMES = ["yolo_last"]
elif args.models_set == 'occft_yolo':
    MODEL_NAMES = ["occft_yolo"]
else:
    raise ValueError("Invalid --models_set argument. Use 'occft' for occluded fine-tuned models, 'federica' for Federica's models.")

# Check if all model paths exist
for model_name in MODEL_NAMES:
    model_path = ALL_MODELS_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")


TEST_SET_H5_PATH = TEST_SET_PATHS[args.test_set]['h5_path']
TEST_SET_IMAGES_PATH = TEST_SET_PATHS[args.test_set]['images_path']

if not os.path.exists(TEST_SET_H5_PATH):
    raise FileNotFoundError(f"Test set file not found: {TEST_SET_H5_PATH}")
if not os.path.exists(TEST_SET_IMAGES_PATH):
    raise FileNotFoundError(f"Test set images folder not found: {TEST_SET_IMAGES_PATH}")

# OUTPUT FOLDER
if args.output_folder:
    OUTPUT_BASE_FOLDER_PATH = args.output_folder
else:
    OUTPUT_BASE_FOLDER_PATH = SAVED_IMAGES_PATH

if args.quick:
    BATCH_SIZE = 3  # Process only 3 images in quick mode
    print(f"[WARNING] Running in QUICK mode: only a small subset of the test data will be processed. Also setting batch size to {BATCH_SIZE} as we only run one batch")
else:
    BATCH_SIZE = 64

run_name = f"{get_timestamp()}_extpert"
run_name += "_quick-run" if args.quick else "_cmplt-run"
run_name += f"_{args.models_set}-models"
run_name += f"_{args.test_set}-testset"

# Redirect output if specified
if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{run_name}.log")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH)

print(f"========== SETTINGS ==========")
print(f"ARGS:")
print(f"\t--quick: {args.quick}")
print(f"\t--redirect_output: {args.redirect_output}")
print(f"\t--models_set: {args.models_set}")
print(f"\t--test_set: {args.test_set}")
print(f"\t--square_sizes: {args.square_sizes}")
print(f"\t--blur_sigma: {args.blur_sigma}")
print(f"CONSTANTS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET_PATH: {TEST_SET_H5_PATH}")
print(f"\tTEST_SET_IMAGES_PATH: {TEST_SET_IMAGES_PATH}")
print(f"\tOUTPUT_BASE_FOLDER_PATH: {OUTPUT_BASE_FOLDER_PATH}")
print(f"\tRUN_NAME: {run_name}")
if args.redirect_output:
    print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"\tBATCH_SIZE: {BATCH_SIZE}")
print(f"==============================")


# ==================================== MAIN ====================================

# Example usage:
# >>> test run
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occluded --quick --redirect_output
# >>> test run yolo
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft_yolo --test_set occluded --quick --redirect_output


# # >>> occft on subsets
# # occluded-matching
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occluded-matching --redirect_output

# # occluded-mismatching
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occluded-mismatching --redirect_output

# # occlusion-positive-angry
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-angry --redirect_output

# # occlusion-positive-disgust
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-disgust --redirect_output

# # occlusion-positive-fear
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-fear --redirect_output

# # occlusion-positive-happy
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-happy --redirect_output

# # occlusion-positive-sad
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-sad --redirect_output

# # occlusion-positive-surprise
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-positive-surprise --redirect_output

# # occlusion-negative-angry
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-angry --redirect_output

# # occlusion-negative-disgust
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-disgust --redirect_output

# # occlusion-negative-fear
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-fear --redirect_output

# # occlusion-negative-happy
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-happy --redirect_output

# # occlusion-negative-sad
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-sad --redirect_output

# # occlusion-negative-surprise
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_extpert_keras.py" --models_set occft --test_set occlusion-negative-surprise --redirect_output
if __name__ == "__main__":
    output_run_path = os.path.join(OUTPUT_BASE_FOLDER_PATH, run_name)
    os.makedirs(output_run_path, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"[INFO] Processing model: {model_name}")
        model = load_model(model_name)
        test_generator, test_paths = load_test_generator(TEST_SET_H5_PATH, batch_size=BATCH_SIZE, small_subset=args.quick, include_paths=True)

        if "yolo" in model_name.lower():
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predicted_probabilities = []
            for emotion in EMOTIONS:
                results = model.predict(os.path.join(TEST_SET_IMAGES_PATH, emotion), device=device)
                predicted_probabilities.extend([result.probs.data.cpu().numpy() for result in results])
        else:
            predicted_probabilities = model.predict(test_generator.x_data)

        for i, (image_array, gt_probabilities_i, predicted_probabilities_i, path) in tqdm(enumerate(zip(test_generator.x_data, test_generator.y_data, predicted_probabilities, test_paths)), total=len(test_generator.x_data), desc="Processing images"):
            image_name = make_xai_result_image_name(path)

            # show image
            if args.quick:
                Image.fromarray(image_array.astype(np.uint8)).save(os.path.join(output_run_path, f"example_input_image.png"))
                print(f"[DEBUG] Saved example input image to {os.path.join(output_run_path, f'example_input_image.png')}")

            gt = np.argmax(gt_probabilities_i)
            image_probability = predicted_probabilities_i[gt]

            perturbed_images = create_black_square_images(image_array, args.square_sizes)
            saliency_map = calculate_saliency_map(model, model_name, image_array, perturbed_images, image_probability, gt)
            blurred_map = blur_saliency_map(saliency_map, args.blur_sigma)

            output_folder = os.path.join(output_run_path, model_name, f"{EMOTIONS[gt]}")
            os.makedirs(output_folder, exist_ok=True)
            filename_abspath_nonpy = os.path.join(output_folder, image_name)
            save_saliency_map(saliency_map, filename_abspath_nonpy)
            blurred_filaname_abspath_nonpy = filename_abspath_nonpy + "_blurred"
            save_saliency_map(blurred_map, blurred_filaname_abspath_nonpy)
            if args.quick:
                print(f"[DEBUG] Saved saliency map for image {i} ({image_name}) (GT: {EMOTIONS[gt]}, Predicted Prob: {image_probability:.4f}) to {filename_abspath_nonpy}.npy and .png")

        print(f"[INFO] Saliency maps for model {model_name} saved to {output_run_path}")