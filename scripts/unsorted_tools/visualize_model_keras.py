import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model as KModel
from tensorflow.keras.applications import MobileNet, ResNet50V2, VGG19, EfficientNetB1, InceptionV3, ConvNeXtBase

from modules.model import load_model
from modules.config import OCCFT_MODELS_PATHS, IMAGES_SHAPE, MODEL_DOCS_FOLDER
from modules.misc import Tee


LOG_FILE_PATH = os.path.join(MODEL_DOCS_FOLDER, "model_visualization_console_output.txt")
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Tee(LOG_FILE_PATH)
sys.stderr = Tee(LOG_FILE_PATH) 

print("================================== SETTINGS ==================================", flush=True)



if __name__ == "__main__":
    
    for model_name in OCCFT_MODELS_PATHS:
        # --- plot the custom / improved model (existing behavior) ---
        model = load_model(model_name)
        # model.summary()  # print model summary to console
        # save_path = os.path.join(visual_dir, f"pattlited_{model_name}_{trainable_count}.png")
        # plot_model(
        #     model,
        #     to_file=save_path,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True,
        #     dpi=96
        # )

        # --- build & plot the original tf.keras.applications backbone (full model) ---
        original_model = None
        try:
            if "efficientnet" in model_name.lower():
                original_model = EfficientNetB1(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

            elif "vgg" in model_name.lower():
                original_model = VGG19(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

            elif "pattlite" in model_name.lower():
                original_model = MobileNet(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

            elif "resnet" in model_name.lower():
                original_model = ResNet50V2(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

            elif "convnext" in model_name.lower():
                original_model = ConvNeXtBase(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

            elif "inception" in model_name.lower():
                original_model = InceptionV3(input_shape=IMAGES_SHAPE, include_top=False, weights='imagenet')

        except Exception as e:
            print(f"Could not instantiate original application model for {model_name}: {e}")
        
        if original_model is not None:
            # compute trainable params for the original backbone
            original_model.summary()
            # orig_trainable_count = sum(tf.keras.backend.count_params(w) for w in original_model.trainable_weights)
            # original_model_save_filename = os.path.join(visual_dir, f"original_{model_name}_{orig_trainable_count}.png")
            # plot_model(
            #     original_model,
            #     to_file=original_model_save_filename,
            #     show_shapes=True,
            #     show_layer_names=True,
            #     expand_nested=True,
            #     dpi=96
            # )