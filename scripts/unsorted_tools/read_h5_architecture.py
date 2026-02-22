import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import ALL_MODELS_PATHS
from modules.model import load_model  # Import your custom load_model function

RUN_LAYERS = True
RUN_COMPLETE = False

if not RUN_LAYERS and not RUN_COMPLETE:
    print("[WARNING] No operation selected. Please set RUN_LAYERS or RUN_COMPLETE to True.")
    sys.exit(0)

for model_name, path in ALL_MODELS_PATHS.items():
    print(f"\n[INFO] Processing model: {model_name}")

    try:
        # Use the custom load_model function to load the model
        model = load_model(model_name)
        print(f"[INFO] Successfully loaded model: {model_name}")

        if RUN_LAYERS:
            # Print layer information
            print(f"[INFO] Model Summary for {model_name}:")
            model.summary()

        if RUN_COMPLETE:
            # Print detailed information about the model
            print(f"[INFO] Model Config for {model_name}:")
            print(model.to_json(indent=2))

    except Exception as e:
        print(f"[ERROR] Failed to load model '{model_name}': {e}")