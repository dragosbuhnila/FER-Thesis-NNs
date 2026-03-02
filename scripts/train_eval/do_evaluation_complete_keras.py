import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse

from modules.misc import get_timestamp, Tee
from modules.model import load_model
from modules.visualize import plot_image;
from modules.data__load import load_test_generator
from modules.evaluate_completely import evaluate_keras_model, evaluate_agreement
from modules.config import ADELE_180ROTATED_TEST_SET_H5_PATH, ADELE_TEST_SET_IMAGES_PATH, OCCLUDED_TEST_SET_H5_PATH, ADELE_TEST_SET_H5_PATH,\
                            ALL_MODELS_PATHS, CONSOLE_OUTPUTS_PATH, EMOTIONS, OCCLUDED_TEST_SET_RESIZED_PATH



# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Evaluate models on a test set by running accuracies, confusion, dimensinality reduction, agreement scores, and other metrics.")
parser.add_argument('--quick',              action='store_true',                    help="Run a quick evaluation on a small subset of the test data (1 batch).")
parser.add_argument('--redirect_output',    action='store_true',                    help="Redirect console output to a log file.")
parser.add_argument('--models_set', type=str, choices=['occft', 'federica'],        help="Specify which set of models to evaluate: 'occft' for occluded fine-tuned models, 'federica' for Federica's models, or 'all' for all available models.")
parser.add_argument('--model_name',  type=str, help="Specify the name of a single model to evaluate (overrides --models_set but model needs to be within set).")
parser.add_argument('--test_set',   type=str, choices=['occluded', 'original', 'original-180'],     help="Specify which test set to use: 'occluded' for the occluded test set, 'original' for the original test set.")
parser.add_argument('--only_agreement', action='store_true', help="Only run agreement evaluation without evaluating individual models. Requires that the individual model evaluations have already been run and their results are available in the expected folders.")
parser.add_argument('--only_evaluation', action='store_true', help="Only run model evaluation without agreement evaluation.")
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
    
# MODEL NAME OVERRIDE
if args.model_name:
    if args.model_name not in MODEL_NAMES:
        raise ValueError(f"Model name '{args.model_name}' not found in the selected models set '{args.models_set}'. Available models: {MODEL_NAMES}")
    MODEL_NAMES = [args.model_name]

    if "yolo" in args.model_name.lower() and args.quick:
        raise ValueError("Quick evaluation is not supported for YOLO models due to the way the data generator and evaluation are implemented (folders). Please run without --quick for YOLO models.")

# ONLY
if args.only_agreement and args.only_evaluation:
    raise ValueError("Cannot use both --only_agreement and --only_evaluation flags at the same time. Please choose one or neither.")

# TEST SET
if args.test_set == 'occluded':
    TEST_SET_H5_PATH = OCCLUDED_TEST_SET_H5_PATH
    TEST_SET_IMAGES_PATH = OCCLUDED_TEST_SET_RESIZED_PATH
elif args.test_set == 'original':
    TEST_SET_H5_PATH = ADELE_TEST_SET_H5_PATH
    TEST_SET_IMAGES_PATH = ADELE_TEST_SET_IMAGES_PATH
elif args.test_set == 'original-180':
    TEST_SET_H5_PATH = ADELE_180ROTATED_TEST_SET_H5_PATH
    TEST_SET_IMAGES_PATH = None  # not implemented yet
else:
    raise ValueError("Invalid --test_set argument. Use 'occluded' for the occluded test set, 'original' for the original test set.")
# check if path exists else raise
if not os.path.exists(TEST_SET_H5_PATH):
    raise FileNotFoundError(f"Test set file not found: {TEST_SET_H5_PATH}")


if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}_do-evaluation-completely-keras.txt")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 



print(f"========== SETTINGS ==========")
print(f"ARGS:")
for arg, value in vars(args).items():
    print(f"\t{arg}: {value}")
print(f"MACROS:")
print(f"\tMODEL_NAMES: {MODEL_NAMES}")
print(f"\tTEST_SET: {TEST_SET_H5_PATH}")
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
# >>> fede models on original test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set federica --test_set original --redirect_output

# >>> occft_yolo on occluded test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --model_name occft_yolo --test_set occluded --redirect_output --only_evaluation
# >>> occft_yolo on original test set:
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --model_name occft_yolo --test_set original --redirect_output --only_evaluation

# >>> only agreement on all!
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_evaluation_complete_keras.py" --models_set occft --test_set occluded --redirect_output --only_agreement

if __name__ == "__main__":
    if len(MODEL_NAMES) == 1 and "yolo" in MODEL_NAMES[0].lower():
        print("Loading test generator for keras models although the model is a YOLO model because agreement uses gts from there and I don't want to write new code")
    else:
        print("Loading test generator for keras models...")
    test_generator = load_test_generator(TEST_SET_H5_PATH, small_subset=args.quick)  # Set to True for quick testing, False for full evaluation
    print("Test generator loaded.")

    testgen_and_folders_dict = {
        "test_generator": test_generator,
        "test_folder": TEST_SET_IMAGES_PATH 
    }

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
        if not args.only_agreement:
            if "yolo" in model_name.lower():
                accuracies, precision_recall_f1, probabilities, y_true, y_pred = evaluate_keras_model(model, TEST_SET_IMAGES_PATH, model_name, run_name=run_name)
            else:
                accuracies, precision_recall_f1, probabilities, y_true, y_pred = evaluate_keras_model(model, test_generator, model_name, run_name=run_name)

    if args.only_evaluation:
        print("Only evaluation flag is set, skipping agreement evaluation.")
        sys.exit(0)


    # Run agreement evaluation after the first evaluation
    print("Running agreement evaluation...")
    evaluate_agreement(models_and_names, testgen_and_folders_dict, run_name=run_name)
    print("Agreement evaluation completed.")