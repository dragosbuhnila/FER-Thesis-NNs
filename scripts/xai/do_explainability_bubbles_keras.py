import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse

from modules.misc import get_timestamp, Tee
from modules.model import load_model
from modules.data__load import load_test_generator
from modules.explainability import generate_bubbles_planes
from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_H5_PATH, ADELE_TEST_SET_H5_PATH, \
                            ALL_MODELS_PATHS, CONSOLE_OUTPUTS_PATH, SAVED_IMAGES_PATH



# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Generate bubble-based explanations for model predictions on a test set.")
parser.add_argument('--quick',              action='store_true',                    help="Run on a small subset of the test data (1 batch).")
parser.add_argument('--redirect_output',    action='store_true',                    help="Redirect console output to a log file.")
parser.add_argument('--models_set',         type=str,   choices=['occft', 'federica'],        help="Specify which set of models to use: 'occft' for occluded fine-tuned models, 'federica' for Federica's models.")
parser.add_argument('--test_set',           type=str,   choices=['occluded', 'original', 'original-180'],   help="Specify which test set to use: 'occluded' for the occluded test set, 'original' for the original test set.")
parser.add_argument('--iterations',         type=int,   default=200,                help="Number of iterations for bubble generation.")
parser.add_argument('--bubble_radius',      type=int,   default=26,                 help="Radius of bubbles in pixels.")
parser.add_argument('--accuracy_target',    type=float, default=0.5,                help="Target accuracy threshold for bubble placement.")
parser.add_argument('--accuracy_tolerance', type=float, default=0.3,                help="Tolerance range for accuracy threshold.")
parser.add_argument('--output_folder',      type=str,                               help="Base folder path for saving generated bubbles planes.")
args = parser.parse_args()

# MODELS
if args.models_set == 'occft':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
elif args.models_set == 'federica':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "finetuning" in model_name.lower()]
else:
    raise ValueError("Invalid --models_set argument. Use 'occft' for occluded fine-tuned models, 'federica' for Federica's models.")
# Check if all model paths exist
for model_name in MODEL_NAMES:
    model_path = ALL_MODELS_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

# TEST SET
if args.test_set == 'occluded':
    TEST_SET = OCCLUDED_TEST_SET_H5_PATH
elif args.test_set == 'original':
    TEST_SET = ADELE_TEST_SET_H5_PATH
elif args.test_set == 'original-180':
    TEST_SET = ADELE_180ROTATED_TEST_SET_H5_PATH
else:
    raise ValueError("Invalid --test_set argument. Use 'occluded' for the occluded test set, 'original' for the original test set.")
# Check if path exists
if not os.path.exists(TEST_SET):
    raise FileNotFoundError(f"Test set file not found: {TEST_SET}")

# output_folder
if args.output_folder:
    OUTPUT_PATH = args.output_folder
else:
    OUTPUT_PATH = SAVED_IMAGES_PATH

# Redirect output if specified
if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}_do-generate-bubbles-planes.txt")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH)

print(f"========== SETTINGS ==========")
print(f"ARGS:")
print(f"\t--quick: {args.quick}")
print(f"\t--redirect_output: {args.redirect_output}")
print(f"\t--models_set: {args.models_set}")
print(f"\t--test_set: {args.test_set}")
print(f"\t--iterations: {args.iterations}")
print(f"\t--bubble_radius: {args.bubble_radius}")
print(f"\t--accuracy_target: {args.accuracy_target}")
print(f"\t--accuracy_tolerance: {args.accuracy_tolerance}")
print(f"MACROS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET: {TEST_SET}")
if args.redirect_output:
    print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"==============================")

# =================================================================================================================


# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_explainability_bubbles_keras.py" --models_set occft --test_set occluded --redirect_output --quick
if __name__ == "__main__":
    print("Loading test generator...")
    test_generator = load_test_generator(TEST_SET, small_subset=args.quick)  # Set to True for quick testing, False for full evaluation
    print("Test generator loaded.")

    run_name = f"{get_timestamp()}_bubbles"
    run_name += f"_quick-run" if args.quick else "_cmplt-run"
    run_name += f"_{args.models_set}-models"
    run_name += f"_{args.test_set}-testset"

    for model_name in MODEL_NAMES:
        print(f"[INFO] Generating bubbles for model: {model_name}")
        model = load_model(model_name)

        generate_bubbles_planes(
            model=model,
            model_name=model_name,
            test_generator=test_generator,
            output_base_folder_path=OUTPUT_PATH,
            run_name=run_name,
            iterations=args.iterations,
            bubble_radius=args.bubble_radius,
            accuracy_target=args.accuracy_target,
            accuracy_tolerance=args.accuracy_tolerance
        )

    print("Bubble generation completed.")