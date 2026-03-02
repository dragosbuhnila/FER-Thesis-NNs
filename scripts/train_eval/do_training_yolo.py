import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

import argparse
import mlflow

from modules.config import OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_YOLO_PATH,\
                            OCCLUDED_TEST_SET_RESIZED_PATH,\
                            MLFLOW_DIR, MLFLOW_DB_WINDOWS, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.misc import Tee, get_timestamp
from modules.yolo import evaluate_model_yolo_training_run, load_yolo_model, save_model_yolo_training_run, train_model_yolo_training_run, FREEZING_MODULES_LAYERS



# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TRAIN_VAL_SET_YOLO_PATH = OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_YOLO_PATH
TEST_SET_IMAGES_PATH = OCCLUDED_TEST_SET_RESIZED_PATH

# Training code from notebook:
# results = model.train(data='/content/drive/MyDrive/Colab Notebooks/HPC/finale/dataset', epochs=300, batch=64, imgsz=128, save_period=3,
#                       resume = True,
#                       patience = 30, auto_augment ='autoaugment',
#                       val=True,save_json=True, plots=True,cache=True,
#                       mosaic = 0.0, freeze = 5,
#                       dropout=0.2, lr0=0.001, project='/content/drive/MyDrive/Colab Notebooks/HPC/finale/yolov8n',
#                       name='yolov8n')#Ricorda di inserire il name corretto per la sottocartella

parser = argparse.ArgumentParser(description='Training parameters for occlusion finetuning')
parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=64)
parser.add_argument('--epochs', type=int, required=True, help='Training epochs')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='Learning rate (default: 0.0001)')
parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
parser.add_argument('--freezing_module', type=str, required=True, choices=FREEZING_MODULES_LAYERS.keys(), help=f"Freezing module. Choose from: {list(FREEZING_MODULES_LAYERS.keys())}")
parser.add_argument('--patience', type=int, required=False, default=3, help='Early stopping patience (default: 3)')
parser.add_argument('--redirect_output', action='store_true', help='If set, redirect stdout and stderr to a log file')
parser.add_argument('--quick', action='store_true', help='If set, use a smaller subset of the data for a quick test run')
args = parser.parse_args()

if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__try_training_yolo.txt")

    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 
else: 
    LOG_FILE_PATH = None

if os.name == 'nt':  # Windows
    # Use SQLite database for MLflow tracking on Windows because URI keeps failing wthelly
    sqlite_db_path = MLFLOW_DB_WINDOWS
    tracking_uri = f"sqlite:///{sqlite_db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"try_training_{args.model_name}"
    mlflow.set_experiment(experiment_name)
else:
    tracking_uri = f"file://{os.path.abspath(MLFLOW_DIR)}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"try_training_{args.model_name}"
    mlflow.set_experiment(experiment_name)

if args.quick:
    print("QUICK RUN ENABLED: Using a smaller subset of the data for a quick test run and limiting epochs to 3.")
    args.epochs = min(args.epochs, 3)

print(f"================================ SETTINGS ==================================")
print(f"CONSTANTS: ")
print(f"\tTRAIN_VAL_SET_YOLO_PATH: {TRAIN_VAL_SET_YOLO_PATH}")
print(f"\tTEST_SET_IMAGES_PATH: {TEST_SET_IMAGES_PATH}")
print(f"ARGS: ")
for arg_name, arg_value in vars(args).items():
    print(f"\t{arg_name}: {arg_value}")
print(f"MLFLOW:")
if tracking_uri is not None:
    print(f"\ttracking_uri: {tracking_uri}")
print(f"\texperiment_name: {experiment_name}")
print(f"GLOBALS:")
for key, value in GLOBALS.items():
    print(f"\t{key}: {value}")
print(f"===========================================================================")
print(f"================================ SETTINGS (yes, again) ==================================")
print(f"CONSTANTS: ")
print(f"\tTRAIN_VAL_SET_YOLO_PATH: {TRAIN_VAL_SET_YOLO_PATH}")
print(f"\tTEST_SET_IMAGES_PATH: {TEST_SET_IMAGES_PATH}")
print(f"ARGS: ")
for arg_name, arg_value in vars(args).items():
    print(f"\t{arg_name}: {arg_value}")
print(f"MLFLOW:")
if tracking_uri is not None:
    print(f"\ttracking_uri: {tracking_uri}")
print(f"\texperiment_name: {experiment_name}")
print(f"GLOBALS:")
for key, value in GLOBALS.items():
    print(f"\t{key}: {value}")
print(f"===========================================================================")


# example usage: 
# >>> YOLO hpc test
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training_yolo.py" --model_name yolo_last --batch_size 64 --epochs 10 --dropout_rate 0.3 --freezing_module most_unfrozen --quick
# >>> YOLO local test
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training_yolo.py" --model_name yolo_last --batch_size 8  --epochs 10 --dropout_rate 0.3 --freezing_module most_unfrozen --quick --redirect_output

if __name__ == "__main__":
    training_and_validation_yolo_folder = TRAIN_VAL_SET_YOLO_PATH
    test_folder_path = TEST_SET_IMAGES_PATH

    model = load_yolo_model(args.model_name)

    with mlflow.start_run():
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        run_name = get_timestamp()
        run_name += "_cmplt-run" if not args.quick else "quick-run"
        freezing_module_high_dash = args.freezing_module.replace("_", "-")
        run_name += f"_freeze-{args.freezing_module}"


        train_model_yolo_training_run(model, training_and_validation_yolo_folder,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      freezing_module=args.freezing_module, dropout_rate=args.dropout_rate, patience=args.patience,
                                      learning_rate=args.learning_rate, quick_run=args.quick)

        test_acc = evaluate_model_yolo_training_run(model, test_folder_path, debug=False)

        save_model_yolo_training_run(model, args.model_name, test_acc, run_name)
