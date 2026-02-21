import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.evaluate_completely import evaluate_agreement
from modules.misc import get_timestamp, Tee
from modules.model import load_model
from modules.data__load import load_test_generator
from modules.config import OCCLUDED_TEST_SET_H5_PATH, ALL_MODELS_PATHS, CONSOLE_OUTPUTS_PATH, OCCFT_MODELS_RESULTS_PATHS



MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
TEST_SET = OCCLUDED_TEST_SET_H5_PATH
QUICK_TESTING = False

LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__try_evaluating_agreement.txt")
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Tee(LOG_FILE_PATH)
sys.stderr = Tee(LOG_FILE_PATH) 



import numpy as np
import pandas as pd
import os

EMOTIONS = ["ANGRY","DISGUST","FEAR","HAPPY","NEUTRAL","SAD","SURPRISE"]


if __name__ == "__main__":
    print("Loading test generator...")
    test_generator = load_test_generator(TEST_SET, small_subset=QUICK_TESTING)  # Set to True for quick testing, False for full evaluation
    print("Test generator loaded.")

    models_and_names = {} # dictinoary with model_name: model_object
    for model_name in MODEL_NAMES:
        model = load_model(model_name)
        models_and_names[model_name] = model

    evaluate_agreement(models_and_names, test_generator, run_name=f"{get_timestamp()}_agreement_evaluation")

    # for model_name in MODEL_NAMES:
    #     print(f"{OCCFT_MODELS_RESULTS_PATHS[model_name]}")
