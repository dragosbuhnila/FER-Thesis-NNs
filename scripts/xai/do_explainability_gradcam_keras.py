import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import re
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, SeparableConv2D, DepthwiseConv2D
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

from modules.train_eval_save import evaluate_model
from modules.model import build_new_model, load_model
from modules.data__load import load_test_generator
from modules.misc import get_timestamp, Tee
from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, ADELE_TEST_SET_H5_PATH, ALL_MODELS_PATHS, EMOTIONS, OCCLUDED_TEST_SET_H5_PATH, SAVED_IMAGES_PATH, CONSOLE_OUTPUTS_PATH

# TODO: not sure why it uses this as my model doens't need it, in case try running it with and without and see what happens
class LayerScale(Layer):
    def __init__(self, init_values=1e-06, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
            name='gamma'
        )
        super(LayerScale, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        config = super(LayerScale, self).get_config()
        config.update({
            'init_values': self.init_values,
        })
        return config

def preprocess_image(image_array, target_size=(128, 128)):
    """
    Preprocess the image for Grad-CAM.
    """
    img = Image.fromarray(image_array.astype(np.uint8)).resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def generate_gradcam(model, image_array, class_index, target_layer, gradcam):
    """
    Generate Grad-CAM saliency map for the given image and class index.
    """
    score = CategoricalScore([class_index])
    heatmap = gradcam(score, image_array, penultimate_layer=target_layer)[0]

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    return heatmap


def save_gradcam_map(heatmap, output_path):
    """
    Save the Grad-CAM heatmap as a PNG image and NumPy file.
    """
    np.save(output_path + ".npy", heatmap)
    plt.imsave(output_path + ".png", heatmap, cmap='jet')


def get_last_conv_layer_names_per_block(model, strip_prefixes=None):
    """
    Ottiene i nomi degli ultimi layer convoluzionali di ciascun blocco nel modello,
    rimuovendo specifici prefissi se forniti.

    Args:
        model (tf.keras.Model): Il modello Keras da analizzare.
        strip_prefixes (list of str, optional): Lista di prefissi da rimuovere dai nomi dei layer.

    Returns:
        list of str: Nomi degli ultimi layer convoluzionali di ciascun blocco con i prefissi rimossi.
    """
    if strip_prefixes is None:
        strip_prefixes = []

    # Dizionario per memorizzare l'ultimo layer per ciascun blocco
    last_conv_layers = {}

    # Regex per identificare il blocco (es. 'block1a', 'block2b', ecc.)
    pattern = re.compile(r'^(block\d+[a-z]?)_')

    for layer in model.layers:
        if isinstance(layer, (Conv2D, SeparableConv2D, DepthwiseConv2D)):
            # Estrai il nome del blocco
            match = pattern.match(layer.name)
            if match:
                block_name = match.group(1)  # Es. 'block1a'
                # Aggiorna il dizionario; l'ultimo layer trovato sarà conservato
                last_conv_layers[block_name] = layer.name
            else:
                # Se il layer non appartiene a un blocco specifico, puoi decidere come gestirlo
                # Ad esempio, aggiungerlo direttamente
                last_conv_layers[layer.name] = layer.name

    # Ottieni la lista degli ultimi layer convoluzionali per ciascun blocco
    final_layer_names = list(last_conv_layers.values())

    # Rimuovi i prefissi specificati, se presenti
    for prefix in strip_prefixes:
        final_layer_names = [name.replace(prefix, '') for name in final_layer_names]

    return final_layer_names


def get_full_conv_layer_names(model, strip_prefixes=None):
    """
    Ottiene i nomi completi degli ultimi layer convoluzionali di ciascun blocco nel modello, rimuovendo specifici prefissi.

    Args:
        model (tf.keras.Model): Il modello Keras da analizzare.
        strip_prefixes (list of str, optional): Lista di prefissi da rimuovere dai nomi dei layer.

    Returns:
        list of str: Nomi completi degli ultimi layer convoluzionali di ciascun blocco con i prefissi rimossi.
    """
    if strip_prefixes is None:
        strip_prefixes = []

    conv_layer_names = {}
    all_conv_layers = []

    for layer in model.layers:
        if isinstance(layer, (Conv2D, SeparableConv2D, DepthwiseConv2D)):
            # Trova il blocco a cui appartiene il layer
            match = re.search(r'conv\d+_block\d+', layer.name)
            if match:
                block_name = match.group(0)
                conv_layer_names[block_name] = layer.name
            all_conv_layers.append(layer.name)

    # Ottieni i nomi degli ultimi layer di ciascun blocco
    final_layer_names = []
    seen_blocks = set()
    for layer_name in all_conv_layers:
        match = re.search(r'conv\d+_block\d+', layer_name)
        if match:
            block_name = match.group(0)
            if block_name not in seen_blocks:
                seen_blocks.add(block_name)
                final_layer_names.append(conv_layer_names[block_name])
        else:
            final_layer_names.append(layer_name)

    # Rimuovi i prefissi specificati
    for prefix in strip_prefixes:
        final_layer_names = [name.replace(prefix, '') for name in final_layer_names]

    return final_layer_names


def get_layer_names(model, model_name, strip_prefixes=None):
    if "efficient" in model_name.lower():
        return get_last_conv_layer_names_per_block(model, strip_prefixes)
    else:
        return get_full_conv_layer_names(model, strip_prefixes)




# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Generate Grad-CAM saliency maps.")
parser.add_argument('--quick', action='store_true', help="Run on a small subset of the test data (1 batch).")
parser.add_argument('--redirect_output', action='store_true', help="Redirect console output to a log file.")
parser.add_argument('--models_set', type=str, choices=['occft', 'federica'], help="Specify which set of models to use.")
parser.add_argument('--test_set', type=str, choices=['occluded', 'original', 'original-180'], help="Specify which test set to use.")
parser.add_argument('--output_folder', type=str, help="Base folder path for saving Grad-CAM maps.")
args = parser.parse_args()

# MODELS
if args.models_set == 'occft':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
elif args.models_set == 'federica':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "finetuning" in model_name.lower()]
else:
    raise ValueError("Invalid --models_set argument. Use 'occft' or 'federica'.")

# TEST SETS
if args.test_set == 'occluded':
    TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH
elif args.test_set == 'original':
    TEST_SET_PATH = ADELE_TEST_SET_H5_PATH
elif args.test_set == 'original-180':
    TEST_SET_PATH = ADELE_180ROTATED_TEST_SET_H5_PATH
else:
    raise ValueError("Invalid --test_set argument. Use 'occluded', 'original', or 'original-180'.")

# OUTPUT FOLDER
OUTPUT_BASE_FOLDER_PATH = args.output_folder if args.output_folder else SAVED_IMAGES_PATH

if args.quick:
    BATCH_SIZE = 3  # Process only 3 images in quick mode
    print(f"[WARNING] Running in QUICK mode: only a small subset of the test data will be processed. Also setting batch size to {BATCH_SIZE} as we only run one batch")
else:
    BATCH_SIZE = 64

run_name = get_timestamp()
run_name += "_quick-run" if args.quick else "_cmplt-run"
run_name += f"_{args.models_set}-models"
run_name += f"_{args.test_set}-testset"
run_name += f"_do_explainability_gradcam_keras"

# Redirect output if specified
if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{run_name}_do_explainability_gradcam_keras.log")
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
print(f"CONSTANTS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET_PATH: {TEST_SET_PATH}")
print(f"\tBATCH_SIZE: {BATCH_SIZE}")
print(f"==============================")


# ==================================== MAIN ====================================

# Example usage:
# >>> test run
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --redirect_output --models_set occft --test_set occluded
if __name__ == "__main__":
    output_run_path = os.path.join(OUTPUT_BASE_FOLDER_PATH, run_name)
    os.makedirs(output_run_path, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"[INFO] Processing model: {model_name}")
        model = load_model(model_name, additional_custom_objects={'LayerScale': LayerScale})
        target_layer_names = get_layer_names(model.get_layer('base_model'), model_name)

        if "pattlite" in model_name or "vgg" in model_name:
            target_layer_names = get_layer_names(model, model_name)

            gradcam_model = build_new_model(model, model_name)
            model = gradcam_model
            clone = True
        else: 
            target_layer_names = get_layer_names(model.get_layer('base_model'), model_name)

            gradcam_model = model.get_layer('base_model')
            clone = False

        test_generator = load_test_generator(TEST_SET_PATH, batch_size=BATCH_SIZE, small_subset=args.quick)
        test_loss, test_acc = evaluate_model(model, model_name, test_generator)
        print(f"[INFO] Test accuracy for model {model_name} on test set {args.test_set}: {test_acc:.4f}")
        
        print(f"[DEBUG] Successfully loaded model {model_name}. Target layers ({len(target_layer_names)} layers):")
        for target_layer_name in target_layer_names:
            print(f"\t- {target_layer_name}")

        try:
            gradcam = Gradcam(gradcam_model, model_modifier=ReplaceToLinear(), clone=clone)
        except Exception as e:
            print(f"[ERROR] Failed to initialize GradCAM for model {model_name}: {e}")
            continue


        print(f"[INFO] Found {len(target_layer_names)} target layers for Grad-CAM")
        for i, (image_array, gt_probabilities_i) in tqdm(enumerate(zip(test_generator.x_data, test_generator.y_data)), total=len(test_generator.x_data), desc="Processing images"):
            for target_layer_name in target_layer_names:
                # Save example input image in quick mode
                if args.quick:
                    Image.fromarray(image_array.astype(np.uint8)).save(os.path.join(output_run_path, f"example_input_image_{i}_layer_{target_layer_name}.png"))
                    print(f"[DEBUG] Saved example input image to {os.path.join(output_run_path, f'example_input_image_{i}_layer_{target_layer_name}.png')}")

                gt = np.argmax(gt_probabilities_i)

                # Preprocess the image
                preprocessed_image = preprocess_image(image_array)

                # Generate Grad-CAM saliency map
                heatmap = generate_gradcam(model, preprocessed_image, gt, target_layer_name, gradcam)

                # Save the saliency map
                output_folder = os.path.join(output_run_path, model_name, target_layer_name, f"{EMOTIONS[gt]}")
                os.makedirs(output_folder, exist_ok=True)
                filename_abspath_nonpy = os.path.join(output_folder, f"image_{i}")
                save_gradcam_map(heatmap, filename_abspath_nonpy)

                if args.quick:
                    print(f"[DEBUG] Saved Grad-CAM map for image {i} (GT: {EMOTIONS[gt]}) to {filename_abspath_nonpy}.npy and .png")

        print(f"[INFO] Grad-CAM maps for model {model_name} saved to {output_run_path}")