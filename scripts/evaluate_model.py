import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import tensorflow as tf

from modules.config import  ACCURACY_RESULTS_PATH, ALL_MODELS_PATHS, \
                            ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_YAML_PATH, ADELE_TEST_SET_IMAGES_PATH, \
                            OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_YAML_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH 
from modules.data__load import load_online_test_generator
from modules.model import load_model
from modules.train_eval import evaluate_model



# ============== MACROS ===============

PATHS = {
    "ADELE": {
        "test_set_big": None,
        "test_set_small": ADELE_TEST_SET_IMAGES_PATH,
        "test_set_h5": ADELE_TEST_SET_H5_PATH,
        "test_set_yaml": ADELE_TEST_SET_YAML_PATH,
    },
    "OCCLUDED": {
        "test_set_big": OCCLUDED_TEST_SET_IMAGES_PATH,
        "test_set_small": OCCLUDED_TEST_SET_RESIZED_PATH,
        "test_set_h5": OCCLUDED_TEST_SET_H5_PATH,
        "test_set_yaml": OCCLUDED_TEST_SET_YAML_PATH,
    }
}

available_models = list(ALL_MODELS_PATHS.keys())

# 0) Setup macros as args
parser = argparse.ArgumentParser(description='Evaluate model on test sets')
parser.add_argument('--test_set', nargs='?', required=True, choices=list(PATHS.keys()), help=f'Test set to use for evaluation. Options: {list(PATHS.keys())}')
parser.add_argument('--model_name', type=str, required=True, choices=available_models, help=f'Name of the model to evaluate. Options: {available_models}')

args = parser.parse_args()
TEST_SET = args.test_set
MODEL_NAME = args.model_name


MODEL_PATHS_SUBSET = ALL_MODELS_PATHS
LOG_FILE = os.path.join(ACCURACY_RESULTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}_accuracies_{TEST_SET.lower()}.log")

REDIRECT_OUTPUT = False
DEBUG = False
YOLO_FOLDERS_INSTEAD_OF_GENERATOR = False  # only for YOLO models, to test accuracy issues

if YOLO_FOLDERS_INSTEAD_OF_GENERATOR and "yolo" not in MODEL_NAME.lower():
    raise ValueError("YOLO_FOLDERS_INSTEAD_OF_GENERATOR can only be True for YOLO models, to test accuracy issues using different evaluation methods.")

# =========== END OF MACROS ===========



# ================= Global ==================

if REDIRECT_OUTPUT:
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout

# Print the LD_LIBRARY_PATH environment variable
ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
print(f"LD_LIBRARY_PATH: {ld_library_path}")

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f" - {gpu}")
    # Set memory growth to avoid allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. The code will run on the CPU.")

# =================== End Of Global =====================



# ================= Main ==================
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe c:/Users/Dragos/Roba/Lectures/YM2.2/models_repo_fixing/FER-Thesis-NNs/scripts/evaluate_model.py --test_set OCCLUDED --model_name convnext_finetuning
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe c:/Users/Dragos/Roba/Lectures/YM2.2/models_repo_fixing/FER-Thesis-NNs/scripts/evaluate_model.py --test_set OCCLUDED --model_name occft_convnext
if __name__ == "__main__":
    # 1) Load the test set
    # # if you can't find the h5 file, generate it from the images
    # # ACTUALLY, JUST GENERATE IT BEFORE RUNNING THIS, I DON'T WANT POSSIBLE BUGS FROM THIS
    # if not os.path.exists(PATHS[TEST_SET]["test_set_h5"]):
    #     generate_h5_from_images(PATHS[TEST_SET]["test_set"], PATHS[TEST_SET]["test_set_resized"], PATHS[TEST_SET]["test_set_h5"]
    
    # (path, occlusion_probability, masking_function, mismatch)
    test_generator = load_online_test_generator(PATHS[TEST_SET]["test_set_h5"])

    print(f"Loaded {TEST_SET} test set with {len(test_generator.x_data)} samples.")

    # 2) Run the evaluations on the test set
    models_results = {name: {"test_loss": None, "test_acc": None} for name in [MODEL_NAME]}

    print("======================================")
    print(f"Evaluating model: {MODEL_NAME}")

    # a) Load the model
    model = load_model(MODEL_NAME, MODEL_PATHS_SUBSET, debug=DEBUG)

    if model is None:
        raise ValueError(f"load_model returned None. Model loading not implemented for this model type. Model name: {MODEL_NAME}")
    else:
        # b) Evaluate the model
        if not YOLO_FOLDERS_INSTEAD_OF_GENERATOR or "yolo" not in MODEL_NAME.lower():
            test_loss, test_acc = evaluate_model(model, MODEL_NAME, test_generator, debug=DEBUG)
        else:
            # THIS EXISTS FOR YOLO. FOR NOW THE "CORRECT" VERSION IS THE ONE WITH FOLDERS
            test_loss, test_acc = evaluate_model(model, MODEL_NAME, None, PATHS[TEST_SET]["test_set_small"], debug=DEBUG)
        
        models_results[MODEL_NAME]["test_loss"] = test_loss
        models_results[MODEL_NAME]["test_acc"] = test_acc
    print("======================================")

    # 3) Print the final results
    print(f"\n\nFinal evaluation results on {TEST_SET.lower()} test set:")
    for model_name, results in models_results.items():
        print(f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}")
    