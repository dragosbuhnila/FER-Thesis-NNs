import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

import argparse
import mlflow

from modules.config import OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,\
                            OCCLUDED_TEST_SET_IMAGES_PATH,\
                            MLFLOW_DIR, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.misc import Tee, get_timestamp
from modules.yolo import evaluate_model_yolo_training_run, load_yolo_model, save_model_yolo_training_run, train_model_yolo_training_run



# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TRAIN_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH
VAL_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH
TEST_SET_IMAGES_PATH = OCCLUDED_TEST_SET_IMAGES_PATH



parser = argparse.ArgumentParser(description='Training parameters for occlusion finetuning')
parser.add_argument('--unfreeze', type=int, required=True, help='Number of layers to unfreeze for fine-tuning. If not provided.')
parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization parameter')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
parser.add_argument('--epochs', type=int, required=True, help='Training epochs')
parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=64)
parser.add_argument('--redirect_output', action='store_true', help='If set, redirect stdout and stderr to a log file')
args = parser.parse_args()

# others
OTHER_TRAINING_PARAMS = {
    "TRAIN_ES_PATIENCE": 3,
    "TRAIN_LR_PATIENCE": 2,
    "ES_LR_MIN_DELTA": 0.0001,
    "TRAIN_MIN_LR": 1e-6
}

if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__{__name__}__console_output.txt")

    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 
else: 
    LOG_FILE_PATH = None

tracking_uri = f"file://{os.path.abspath(MLFLOW_DIR)}"
mlflow.set_tracking_uri(tracking_uri)
experiment_name = f"try_training_{args.model_name}"
mlflow.set_experiment(experiment_name)

print(f"================================ SETTINGS ==================================")
print(f"CONSTANTS: ")
print(f"\tTRAIN_SET_IMAGES_PATH: {TRAIN_SET_IMAGES_PATH}")
print(f"\tVAL_SET_IMAGES_PATH: {VAL_SET_IMAGES_PATH}")
print(f"\tTEST_SET_IMAGES_PATH: {TEST_SET_IMAGES_PATH}")
print(f"ARGS: ")
for arg_name, arg_value in vars(args).items():
    print(f"\t{arg_name}: {arg_value}")
print(f"OTHER TRAINING PARAMS:")
for key, value in OTHER_TRAINING_PARAMS.items():
    print(f"\t{key}: {value}")
print(f"MLFLOW:")
print(f"\ttracking_uri: {tracking_uri}")
print(f"\texperiment_name: {experiment_name}")
print(f"GLOBALS:")
for key, value in GLOBALS.items():
    print(f"\t{key}: {value}")
print(f"===========================================================================")



# example usage: 
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training.py" --batch_size 8 --matching_amount 0.2 --val_occ_prob 0.5 --occ_prob 0.2 --model_name ConvNeXt --FT_EPOCH 300  --dropout_rate 0.5 --learning_rate  0.0001 --l2_reg 0.00200041712955451

if __name__ == "__main__":
    training_folder_path = TRAIN_SET_IMAGES_PATH
    val_folder_path = VAL_SET_IMAGES_PATH
    test_folder_path = TEST_SET_IMAGES_PATH

    model = load_yolo_model(args.model_name)

    with mlflow.start_run():
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        train_model_yolo_training_run(model, training_folder_path, val_folder_path,
                                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)

        test_acc = evaluate_model_yolo_training_run(model, test_folder_path, debug=False)

        save_model_yolo_training_run(model, args.model_name, test_acc)
