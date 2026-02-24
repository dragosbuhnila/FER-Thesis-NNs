import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import time
import tensorflow as tf

from modules.config import  ACCURACY_RESULTS_PATH, ALL_MODELS_PATHS, \
                            ADELE_TEST_SET_H5_PATH, ADELE_TEST_SET_YAML_PATH, ADELE_TEST_SET_IMAGES_PATH, \
                            OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_YAML_PATH, OCCLUDED_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, \
                            ADELE_180ROTATED_TEST_SET_H5_PATH, ADELE_180ROTATED_TEST_SET_IMAGES_PATH
from modules.data__load import load_test_generator
from modules.model import load_model
from modules.train_eval_save import valuta_modello, evaluate_model



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
    },
    "180ADELE": {
        "test_set_small": ADELE_180ROTATED_TEST_SET_IMAGES_PATH,
        "test_set_h5": ADELE_180ROTATED_TEST_SET_H5_PATH,
    }
}

available_models = list(ALL_MODELS_PATHS.keys())

# 0) Setup macros as args
parser = argparse.ArgumentParser(description='Evaluate model on test sets')
parser.add_argument('--test_set', choices=list(PATHS.keys()), required=True,    help=f'Test set to use for evaluation. Options: {list(PATHS.keys())}')
parser.add_argument('--model_name', choices=available_models, required=True,    help=f'Name of the model to evaluate. Options: {available_models}')
parser.add_argument('--yolo_use_folders_instead_of_gen', action="store_true",   help=f"If set, it will use an alternative function from Federica's code, which unfortunately yields a different result")
parser.add_argument('--redirect_output', action="store_true",                   help="Redirect the output to a log file. Console will still view output in real time.")
parser.add_argument('--debug', action="store_true",                             help="Print lots of info during run.")

args = parser.parse_args()
TEST_SET = args.test_set
MODEL_NAME = args.model_name

USA_VALUTA_INVECE_DI_EVALUATE = False
# you may choose to enable this option, but if model is detected to be yolo it will automatically be falsified
if "yolo" in MODEL_NAME.lower():
    USA_VALUTA_INVECE_DI_EVALUATE = False

if args.yolo_use_folders_instead_of_gen and "yolo" not in MODEL_NAME.lower():
    raise ValueError("YOLO_FOLDERS_INSTEAD_OF_GENERATOR can only be True for YOLO models, to test accuracy issues using different evaluation methods.")

MODEL_PATHS_SUBSET = ALL_MODELS_PATHS
LOG_FILE = os.path.join(ACCURACY_RESULTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}_accuracies_{TEST_SET.lower()}.log")

# =========== END OF MACROS ===========



# ================= Global ==================

if args.redirect_output:
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
# >>> Running **fede-yolo** on **original-set**
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set ADELE --model_name yolo_last

# >>> Running **fede-yolo** on **original-set** with **folders** instead of gen
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set ADELE --model_name yolo_last --yolo_use_folders_instead_of_gen

# >>> Running **fede-yolo** on **occluded-set**
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set OCCLUDED --model_name yolo_last

# >>> Running **fede-yolo** on **occluded-set** with **folders** instead of gen
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set OCCLUDED --model_name yolo_last --yolo_use_folders_instead_of_gen

# >>> Running **fede-yolo** on **180original-set**
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set 180ADELE --model_name yolo_last

# >>> Running **fede-yolo** on **180original-set** with **folders** instead of gen
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_accuracy_simple.py" --test_set 180ADELE --model_name yolo_last --yolo_use_folders_instead_of_gen
if __name__ == "__main__":
    # 1) Load the test set
    # # if you can't find the h5 file, generate it from the images
    # # ACTUALLY, JUST GENERATE IT BEFORE RUNNING THIS, I DON'T WANT POSSIBLE BUGS FROM THIS
    # if not os.path.exists(PATHS[TEST_SET]["test_set_h5"]):
    #     generate_h5_from_images(PATHS[TEST_SET]["test_set"], PATHS[TEST_SET]["test_set_resized"], PATHS[TEST_SET]["test_set_h5"]
    
    # (path, occlusion_probability, masking_function, mismatch)
    test_generator = load_test_generator(PATHS[TEST_SET]["test_set_h5"])

    print(f"Loaded {TEST_SET} test set with {len(test_generator.x_data)} samples.")

    # 2) Run the evaluations on the test set
    models_results = {name: {"test_loss": None, "test_acc": None} for name in [MODEL_NAME]}

    print("======================================")
    print(f"Evaluating model: {MODEL_NAME}")

    # a) Load the model
    model = load_model(MODEL_NAME, MODEL_PATHS_SUBSET, debug=args.debug)

    if model is None:
        raise ValueError(f"load_model returned None. Model loading not implemented for this model type. Model name: {MODEL_NAME}")
    else:
        # b) Evaluate the model
        if not args.yolo_use_folders_instead_of_gen or "yolo" not in MODEL_NAME.lower():
            if USA_VALUTA_INVECE_DI_EVALUATE:
                test_loss, test_acc = valuta_modello(model, test_generator, None, MODEL_NAME)
            else:
                test_loss, test_acc = evaluate_model(model, MODEL_NAME, test_generator, debug=args.debug)
        else:
            # THIS EXISTS FOR YOLO. FOR NOW THE "CORRECT" VERSION IS THE ONE WITH FOLDERS
            test_loss, test_acc = evaluate_model(model, MODEL_NAME, None, PATHS[TEST_SET]["test_set_small"], debug=args.debug)
        
        models_results[MODEL_NAME]["test_loss"] = test_loss
        models_results[MODEL_NAME]["test_acc"] = test_acc
    print("======================================")

    # 3) Print the final results
    print(f"\n\nFinal evaluation results on {TEST_SET.lower()} test set:")
    for model_name, results in models_results.items():
        msg_to_print = f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}"
        if "yolo" in model_name:
            if args.yolo_use_folders_instead_of_gen:
                msg_to_print += " (used folders instead of testgen)"
            else:
                msg_to_print += " (used testgen)"
        print(msg_to_print)
    