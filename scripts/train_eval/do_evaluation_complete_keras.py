import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse

from modules.misc import get_timestamp, Tee
from modules.model import load_model
from modules.visualize import plot_image;
from modules.data__load import load_test_generator
from modules.evaluate_completely import evaluate_keras_model, evaluate_agreement
from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_H5_PATH, ADELE_TEST_SET_H5_PATH,\
                            ALL_MODELS_PATHS, CONSOLE_OUTPUTS_PATH, EMOTIONS



# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Evaluate models on a test set by running accuracies, confusion, dimensinality reduction, agreement scores, and other metrics.")
parser.add_argument('--quick',              action='store_true',                    help="Run a quick evaluation on a small subset of the test data (1 batch).")
parser.add_argument('--redirect_output',    action='store_true',                    help="Redirect console output to a log file.")
parser.add_argument('--models_set', type=str, choices=['occft', 'federica'],        help="Specify which set of models to evaluate: 'occft' for occluded fine-tuned models, 'federica' for Federica's models, or 'all' for all available models.")
parser.add_argument('--test_set',   type=str, choices=['occluded', 'original', 'original-180'],     help="Specify which test set to use: 'occluded' for the occluded test set, 'original' for the original test set.")
args = parser.parse_args()

# MODELS
if args.models_set == 'occft':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "occft" in model_name.lower()]
elif args.models_set == 'federica':
    MODEL_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if "finetuning" in model_name.lower()]
else:
    raise ValueError("Invalid --models_set argument. Use 'occft' for occluded fine-tuned models, 'federica' for Federica's models.")
# check if all model paths exist else raise
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
# check if path exists else raise
if not os.path.exists(TEST_SET):
    raise FileNotFoundError(f"Test set file not found: {TEST_SET}")


if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}_do-evaluation-completely-keras.txt")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 



print(f"========== SETTINGS ==========")
print(f"ARSG:")
print(f"\t--quick: {args.quick}")
print(f"\t--redirect_output: {args.redirect_output}")
print(f"\t--models_set: {args.models_set}")
print(f"\t--test_set: {args.test_set}")
print(f"MACROS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET: {TEST_SET}")
if args.redirect_output:
    print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"==============================")

# =================================================================================================================

# >>> test run:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --test_set occluded --redirect_output --quick
# >>> occft run on occluded test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --test_set occluded
# >>> occft run on original test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --test_set original
# >>> federica run on 180rotated original test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set federica --test_set original-180

if __name__ == "__main__":
    print("Loading test generator...")
    test_generator = load_test_generator(TEST_SET, small_subset=args.quick)  # Set to True for quick testing, False for full evaluation
    print("Test generator loaded.")

    # for batch in test_generator:
    #     for image, label_probs in zip(batch[0], batch[1]):
    #         label = label_probs.argmax()
    #         plot_image(image, f"True Label: {EMOTIONS[label]} (idx={label})")

    run_name = f"{get_timestamp()}"
    run_name += f"_quick-run" if args.quick else "_cmplt-run"
    run_name += f"_{args.models_set}-models"
    run_name += f"_{args.test_set}-testset"
    run_name += "_do-evaluation-completely-keras"

    models_and_names = {}  # Dictionary to store model_name: model_object
    for model_name in MODEL_NAMES:
        print(f"Evaluating model: {model_name}")
        model = load_model(model_name)
        models_and_names[model_name] = model

        # Evaluate the model
        accuracies, precision_recall_f1, probabilities, y_true, y_pred = evaluate_keras_model(model, test_generator, model_name, run_name=run_name)

    # Run agreement evaluation after the first evaluation
    print("Running agreement evaluation...")
    evaluate_agreement(models_and_names, test_generator, run_name=run_name)
    print("Agreement evaluation completed.")