import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re

from modules.config import GRADCAM_OCC_OCC_DIR_PATH, OCCFT_MODELS_PATHS

def natural_sort_key(s):
    """Generate a key for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    
    model_folders = os.listdir(GRADCAM_OCC_OCC_DIR_PATH)
    for model_folder in model_folders:
        if not model_folder in OCCFT_MODELS_PATHS:
            print(f"Model folder '{model_folder}' not found in OCCFT_MODELS_PATHS. Skipping.")
            continue
        model_folder_path = os.path.join(GRADCAM_OCC_OCC_DIR_PATH, model_folder)
        gradcam_layer_folders = os.listdir(model_folder_path)

        # first round to check if layer_0 folder exists so that you name layers in order correctly
        start_from_layer_1 = False
        for gradcam_layer_folder in gradcam_layer_folders:
            if "layer_0" in gradcam_layer_folder:
                start_from_layer_1 = True
                break

        gradcam_layer_folders = [f for f in gradcam_layer_folders if not "layer_0" in f]
        gradcam_layer_folders.sort(key=natural_sort_key)  # Use natural sorting
        for i, gradcam_layer_folder in enumerate(gradcam_layer_folders):
            if start_from_layer_1:
                new_layer_folder_name = f"{gradcam_layer_folder}_layer_{i+1}"
            else:
                new_layer_folder_name = f"{gradcam_layer_folder}_layer_{i}"

            old_layer_folder_path = os.path.join(model_folder_path, gradcam_layer_folder)
            new_layer_folder_path = os.path.join(model_folder_path, new_layer_folder_name)
            os.rename(old_layer_folder_path, new_layer_folder_path)
            print(f"Renamed to '{new_layer_folder_path}'")