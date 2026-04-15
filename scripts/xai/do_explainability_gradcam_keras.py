import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import re
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from torchvision import transforms
from tensorflow.keras.layers import Conv2D, Layer, SeparableConv2D, DepthwiseConv2D
from tf_keras_vis.gradcam import Gradcam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

from modules.train_eval_save import evaluate_model
from modules.model import build_new_model, load_model
from modules.data__load import load_test_generator
from modules.misc import get_timestamp, Tee, make_xai_result_image_name
from modules.misc import TEST_SET_CHOICES, TEST_SET_PATHS
from modules.config import (
    ALL_MODELS_PATHS,
    CONSOLE_OUTPUTS_PATH,
    SAVED_IMAGES_PATH,
    EMOTIONS,
)



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

def preprocess_image_keras(image_array, target_size=(128, 128)):
    """
    Preprocess the image for Grad-CAM.
    """
    img = Image.fromarray(image_array.astype(np.uint8)).resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_yolo(image_array):
    img = image_array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)
    img.requires_grad = True  # Abilita il calcolo dei gradienti per Grad-CAM
    return img


def generate_gradcam_keras(image_array, class_index, target_layer, gradcam):
    """
    Generate Grad-CAM saliency map for the given image and class index.
    """
    score = CategoricalScore([class_index])
    heatmap = gradcam(score, image_array, penultimate_layer=target_layer)[0]

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    return heatmap


def generate_gradcam_yolo(image, target_label, target_layer, model):
    """
    Generate Grad-CAM saliency map for YOLO model.
    """
    import torch

    # Inizializza GradCAM
    cam = GradCAM(model=model.model, target_layers=target_layer)
    # Prepara il target: il Grad-CAM richiede esplicitamente il target della classe
    targets = [ClassifierOutputTarget(target_label)]  # Classe 0 (modifica in base al tuo caso)

    # Genera la mappa di salienza
    grayscale_cam = cam(input_tensor=image, targets = targets)[0, :]
    #Normalizzazione
    #--------------------------------------------------------
    #normalized_cam = cv2.normalize(grayscale_cam, None, 0, 1, cv2.NORM_MINMAX)
    # Colora la mappa usando una mappa di colori (jet)
    #heatmap = cv2.applyColorMap((normalized_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #--------------------------------------------------------

    #heatmap = cv2.applyColorMap((grayscale_cam).astype(np.uint8), cv2.COLORMAP_JET) #senza sriportare i valori tra 0 e 255
    #heatmap = cv2.cvtColor(normalized_cam, cv2.COLOR_GRAY2RGB)
    heatmap = tf.maximum(grayscale_cam, 0) / tf.math.reduce_max(grayscale_cam)
    # Mappatura dei colori usando colormap di Matplotlib
    #heatmap_rgb = plt.get_cmap('jet')(heatmap)  # Puoi scegliere un colormap diverso, come 'viridis', 'plasma', etc.

    # Rimuovi l'alpha channel (il quarto canale) per ottenere solo RGB
    #heatmap_rgb = heatmap_rgb[..., :3]  # Adesso è una mappa di colori RGB
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
parser.add_argument('--models_set', type=str, choices=['occft', 'federica', 'yolo_fede', 'occft_yolo'], help="Specify which set of models to use.")
parser.add_argument('--model_name', type=str, help="Specify a single model name to process.")
parser.add_argument('--test_set', type=str, choices=TEST_SET_CHOICES, help="Specify which test set to use.")
parser.add_argument('--output_folder', type=str, help="Base folder path for saving Grad-CAM maps.")
parser.add_argument('--no_layer_scale', action='store_true', help="Do not use LayerScale in the model.")
parser.add_argument('--sequential', action='store_true', help="Force the use of the sequential model wrapper for Grad-CAM, if models are pattlite or vgg.")
parser.add_argument('--only_show_layer_names', action='store_true', help="Only print the target layer names for Grad-CAM and exit, without generating saliency maps.")
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

# SINGLE MODEL
if args.model_name:
    if args.model_name in MODEL_NAMES:
        MODEL_NAMES = [args.model_name]
    else:
        raise ValueError(f"Model name '{args.model_name}' not found in the selected models set '{args.models_set}'. Available models: {MODEL_NAMES}")

TEST_SET_H5_PATH = TEST_SET_PATHS[args.test_set]['h5_path']
TEST_SET_IMAGES_PATH = TEST_SET_PATHS[args.test_set]['images_path']

if not os.path.exists(TEST_SET_H5_PATH):
    raise FileNotFoundError(f"Test set file not found: {TEST_SET_H5_PATH}")
if not os.path.exists(TEST_SET_IMAGES_PATH):
    raise FileNotFoundError(f"Test set images folder not found: {TEST_SET_IMAGES_PATH}")

# SEQUENTIAL
if args.sequential:
    if not "pattlite" in MODEL_NAMES[0].lower() and not "vgg" in MODEL_NAMES[0].lower():
        print(f"[WARNING] --sequential flag is set but the selected model(s) do not seem to be pattlite or vgg models. This flag will be ignored.")

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

RUN_NAME = f"{get_timestamp()}_gradcam"
RUN_NAME += "_quick-run" if args.quick else "_cmplt-run"
RUN_NAME += f"_{args.models_set}-models"
RUN_NAME += f"_{args.test_set}-testset"

# Redirect output if specified
if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{RUN_NAME}.log")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH)

print(f"========== SETTINGS ==========")
print(f"ARGS:")
for arg_name, arg_value in vars(args).items():
    print(f"\t{arg_name}: {arg_value}")
print(f"CONSTANTS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET_PATH: {TEST_SET_H5_PATH}")
print(f"\tBATCH_SIZE: {BATCH_SIZE}")
print(f"\tOUTPUT_BASE_FOLDER_PATH: {OUTPUT_BASE_FOLDER_PATH}")
print(f"\tRUN_NAME: {RUN_NAME}")
print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH if args.redirect_output else 'No log file, output not redirected'}")
print(f"==============================")


# ==================================== MAIN ====================================

# Example usage:
# >>> test run keras
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --redirect_output --models_set occft --test_set occluded --sequential
# >>> test run yolo
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --quick --redirect_output --models_set occft_yolo --test_set occluded

# >>> show layer names occft
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occluded --only_show_layer_names --redirect_output

# >>> pattlite only
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occluded --model_name occft_pattlite
# >>> vgg only
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occluded --model_name occft_vgg19
# >>> show layer names federica
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set federica --test_set original --only_show_layer_names


# # >>> occft on subsets
# # occluded-matching
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occluded-matching --redirect_output

# # occluded-mismatching
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occluded-mismatching --redirect_output

# # occlusion-positive-angry
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-angry --redirect_output

# # occlusion-positive-disgust
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-disgust --redirect_output

# # occlusion-positive-fear
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-fear --redirect_output

# # occlusion-positive-happy
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-happy --redirect_output

# # occlusion-positive-sad
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-sad --redirect_output

# # occlusion-positive-surprise
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-positive-surprise --redirect_output

# # occlusion-negative-angry
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-angry --redirect_output

# # occlusion-negative-disgust
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-disgust --redirect_output

# # occlusion-negative-fear
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-fear --redirect_output

# # occlusion-negative-happy
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-happy --redirect_output

# # occlusion-negative-sad
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-sad --redirect_output

# # occlusion-negative-surprise
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_explainability_gradcam_keras.py" --models_set occft --test_set occlusion-negative-surprise --redirect_output
if __name__ == "__main__":
    output_run_path = os.path.join(OUTPUT_BASE_FOLDER_PATH, RUN_NAME)
    os.makedirs(output_run_path, exist_ok=True)

    for model_name in MODEL_NAMES:
        # 0) Load the model
        if args.no_layer_scale:
            print(f"[INFO] Loading model {model_name} without LayerScale")
            model = load_model(model_name)
        else:            
            print(f"[INFO] Loading model {model_name} with LayerScale")
            model = load_model(model_name, additional_custom_objects={'LayerScale': LayerScale})
        # _______________________________________________________________________________________

        # 1) Get target layer names for Grad-CAM, convert to sequential model if needed, and setup the gradcam task
        if ("pattlite" in model_name or "vgg" in model_name) and args.sequential:
            print(f"[INFO] Model {model_name} is a pattlite or vgg model and --sequential flag is set, using sequential model wrapper for Grad-CAM")
            target_layers = get_layer_names(model, model_name)

            sequential_model = build_new_model(model, model_name)
            model = sequential_model
            gradcam_model = sequential_model
            clone = True
        elif "yolo" in model_name.lower():
            print(f"[INFO] Model {model_name} is a YOLO model, using the base model for Grad-CAM")
            target_layers = {
                "module_1": [model.model.model[1]],
                "module_2": [model.model.model[2].cv1, model.model.model[2].cv2],
                "module_3": [model.model.model[3]],
                "module_4": [model.model.model[4].cv1, model.model.model[4].cv2],
                "module_5": [model.model.model[5]],
                "module_6": [model.model.model[6].cv1, model.model.model[6].cv2],
                "module_7": [model.model.model[7]],
                "module_8": [model.model.model[8].cv1, model.model.model[8].cv2],
            }
        else:
            print(f"[INFO] Model {model_name} is not a pattlite or vgg model or --sequential flag is not set, using non-sequential model wrapper for Grad-CAM")
            target_layers = get_layer_names(model.get_layer('base_model'), model_name)

            gradcam_model = model.get_layer('base_model')
            clone = False

        print(f"[DEBUG] Successfully loaded model {model_name}. Target layers ({len(target_layers)} layers):")
        if "yolo" in model_name.lower():
            for module_name, layer_list in target_layers.items():
                print(f"\t- {module_name}:")
                for layer in layer_list:
                    print(f"\t\t- {layer}")
        else:
            for target_layer in target_layers:
                print(f"\t- {target_layer}")

        if args.only_show_layer_names:
            print(f"[INFO] --only_show_layer_names flag is set, exiting after printing target layer names.")
            continue
        
        if "yolo" in model_name.lower():
            # cam = GradCAM(model=model.model, target_layers=target_layers) 
            # For yolo you need to initialize it later bcse you need to specify the target layer(s) already
            print(f"[INFO] Not yet setting up Grad-CAM for YOLO model {model_name} as it requires a different approach and needs to be initialized later with the target layers.")
            gradcam = None
        else:
            print(f"[INFO] Setting up Grad-CAM for model {model_name} with clone={clone}")
            gradcam = Gradcam(gradcam_model, model_modifier=ReplaceToLinear(), clone=clone)
        # _______________________________________________________________________________________

        # 2) Evaluate the model on the test set to check everything is working fine before generating Grad-CAM maps (for now check manually)
        test_generator, test_paths = load_test_generator(TEST_SET_H5_PATH, batch_size=BATCH_SIZE, small_subset=args.quick, include_paths=True)
        if "yolo" in model_name.lower():
            _, test_acc = evaluate_model(model, model_name, None, TEST_SET_IMAGES_PATH)
        else:
            _, test_acc = evaluate_model(model, model_name, test_generator)
            print(f"[INFO] Test accuracy for model {model_name} on test set {args.test_set}: {test_acc:.4f}")
        # _______________________________________________________________________________________


        print(f"[INFO] Found {len(target_layers)} target layers for Grad-CAM")
        for i, (image_array, gt_probabilities_i, path) in tqdm(enumerate(zip(test_generator.x_data, test_generator.y_data, test_paths)), total=len(test_generator.x_data), desc="Processing images"):
            image_name = make_xai_result_image_name(path)

            for target_layer_name in target_layers:
                # Save example input image in quick mode
                if args.quick:
                    Image.fromarray(image_array.astype(np.uint8)).save(os.path.join(output_run_path, f"example_input_{image_name}_layer_{target_layer_name}.png"))
                    print(f"[DEBUG] Saved example input image to {os.path.join(output_run_path, f'example_input_{image_name}_layer_{target_layer_name}.png')}")

                gt = np.argmax(gt_probabilities_i)

                # Preprocess the image
                if "yolo" in model_name.lower():
                    preprocessed_image = preprocess_image_yolo(image_array)
                else:
                    preprocessed_image = preprocess_image_keras(image_array)

                # Generate Grad-CAM saliency map
                if "yolo" in model_name.lower():
                    if gradcam is not None and i == 0:
                        raise ValueError(f"Grad-CAM for YOLO model {model_name} should not have been initialized before, but it is not None. Please check the code.")
                    heatmap = generate_gradcam_yolo(preprocessed_image, gt, target_layers[target_layer_name], model)
                else:
                    heatmap = generate_gradcam_keras(preprocessed_image, gt, target_layer_name, gradcam)

                # Save the saliency map
                output_folder = os.path.join(output_run_path, model_name, target_layer_name, f"{EMOTIONS[gt]}")
                os.makedirs(output_folder, exist_ok=True)
                filename_abspath_nonpy = os.path.join(output_folder, image_name)
                save_gradcam_map(heatmap, filename_abspath_nonpy)

                if args.quick:
                    print(f"[DEBUG] Saved Grad-CAM map for image {i} ({image_name}) (GT: {EMOTIONS[gt]}) to {filename_abspath_nonpy}.npy and .png")

        print(f"[INFO] Grad-CAM maps for model {model_name} saved to {output_run_path}")