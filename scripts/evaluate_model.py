import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import tensorflow as tf

from modules.config import  ACCURACY_RESULTS_PATH, ALL_MODELS_PATHS, \
                            ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_YAML_PATH, ADELE_TEST_SET_IMAGES_PATH, \
                            OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_YAML_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH 
from modules.data import load_test_generator
from modules.model import load_model
from modules.eval import evaluate_model



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

MODEL_PATHS_SUBSET = ALL_MODELS_PATHS
TEST_SET = "ADELE"  # Options: "ADELE", "OCCLUDED"
# MODELS_NAMES = ["resnet_finetuning", "pattlite_finetuning", "vgg19_finetuning", "inceptionv3_finetuning", "convnext_finetuning", "efficientnet_finetuning", "yolo_last"]
MODELS_NAMES = ["yolo_last"]

REDIRECT_OUTPUT = False
LOG_FILE = os.path.join(ACCURACY_RESULTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}_accuracies_{TEST_SET.lower()}.log")

DEBUG = True
YOLO_FOLDERS_INSTEAD_OF_GENERATOR = False  # only for YOLO models, to test accuracy issues

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

if __name__ == "__main__":
    # 0) Setup macros as args
    if sys.argv.__len__() == 2:
        TEST_SET = sys.argv[1]
        if TEST_SET not in PATHS.keys():
            print(f"Unknown TEST_SET: {TEST_SET}. Available options: {list(PATHS.keys())}")
            sys.exit(1)
    elif sys.argv.__len__() > 2:
        print("Usage: python evaluate_model.py [TEST_SET]")
        sys.exit(1)

    # 1) Load the test set
    # # if you can't find the h5 file, generate it from the images
    # # ACTUALLY, JUST GENERATE IT BEFORE RUNNING THIS, I DON'T WANT POSSIBLE BUGS FROM THIS
    # if not os.path.exists(PATHS[TEST_SET]["test_set_h5"]):
    #     generate_h5_from_images(PATHS[TEST_SET]["test_set"], PATHS[TEST_SET]["test_set_resized"], PATHS[TEST_SET]["test_set_h5"])
    test_generator = load_test_generator(PATHS[TEST_SET]["test_set_h5"], 'test', 0.0, False)

    print(f"Loaded {TEST_SET} test set with {len(test_generator.x_data)} samples.")

    # 2) Run the evaluations on the test set
    models_results = {name: {"test_loss": None, "test_acc": None} for name in MODELS_NAMES}

    for model_name in MODELS_NAMES:
        print("======================================")
        print(f"Evaluating model: {model_name}")

        if DEBUG:
            print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

        # a) Load the model
        model = load_model(model_name, MODEL_PATHS_SUBSET, debug=DEBUG)
        if DEBUG:
            print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
        if model is None:
            print("Model loading not implemented for this model type.")
            continue
        else:
            # b) Evaluate the model
            if not YOLO_FOLDERS_INSTEAD_OF_GENERATOR or "yolo" not in model_name:
                test_loss, test_acc = evaluate_model(model, model_name, test_generator, debug=DEBUG)
            else:
                # THIS EXISTS FOR YOLO. FOR NOW THE "CORRECT" VERSION IS THE ONE WITH FOLDERS
                test_loss, test_acc = evaluate_model(model, model_name, None, PATHS[TEST_SET]["test_set_small"], debug=DEBUG)
            
            models_results[model_name]["test_loss"] = test_loss
            models_results[model_name]["test_acc"] = test_acc
    print("======================================")

    # 3) Print the final results
    print(f"\n\nFinal evaluation results on {TEST_SET.lower()} test set:")
    for model_name, results in models_results.items():
        print(f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}")
    