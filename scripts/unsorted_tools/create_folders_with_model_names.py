import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.config import CONFUSION_OCC_OCC_MATRICES_DIR_PATH, OCCFT_MODELS_PATHS


if __name__ == "__main__":
    for model in OCCFT_MODELS_PATHS:
        model_folder_path = os.path.join(CONFUSION_OCC_OCC_MATRICES_DIR_PATH, model)
        os.makedirs(model_folder_path, exist_ok=True)
        print(f"Created folder for model '{model}': {model_folder_path}")