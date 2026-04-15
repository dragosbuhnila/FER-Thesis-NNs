import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.misc import get_timestamp, Tee
from modules.model import load_model
from modules.visualize import plot_image;
from modules.data__load import load_test_generator
from modules.evaluate_completely import evaluate_model_completely
from modules.config import OCCLUDED_TEST_SET_H5_PATH, ALL_MODELS_PATHS, CONSOLE_OUTPUTS_PATH



MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
TEST_SET = OCCLUDED_TEST_SET_H5_PATH
QUICK_TESTING = False

LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__try_evaluating_completely.txt")
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Tee(LOG_FILE_PATH)
sys.stderr = Tee(LOG_FILE_PATH) 



print(f"========== SETTINGS ==========")
print(f"MODEL_NAMES: {MODEL_NAMES}")
print(f"TEST_SET: {TEST_SET}")
print(f"QUICK_TESTING: {QUICK_TESTING}")
print(f"LOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"==============================")



if __name__ == "__main__":
    print("Loading test generator...")
    
    test_generator = load_test_generator(TEST_SET, small_subset=QUICK_TESTING)  # Set to True for quick testing, False for full evaluation

    print("Test generator loaded.")

    # for batch in test_generator:
    #     for image, label_probs in zip(batch[0], batch[1]):
    #         label = label_probs.argmax()
    #         plot_image(image, f"True Label: {EMOTIONS[label]} (idx={label})")

    run_name = f"{get_timestamp()}" if not QUICK_TESTING else f"{get_timestamp()}_quick_testing"
    run_name += "_evaluating_completely"

    for model_name in MODEL_NAMES:
        model = load_model(model_name)
        accuracies, precision_recall_f1, probabilities, y_true, y_pred = evaluate_model_completely(model, test_generator, model_name, run_name=run_name)
        