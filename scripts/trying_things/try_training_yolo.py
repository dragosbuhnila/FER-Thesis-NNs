import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))

import argparse
import mlflow

from modules.config import OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,\
                            OCCLUDED_TEST_SET_IMAGES_PATH,\
                            MLFLOW_DIR, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.misc import Tee, get_timestamp
from modules.yolo import evaluate_model_yolo_training_run, load_yolo_model, save_model_yolo_training_run, train_model_yolo_training_run, FREEZING_MODULES_LAYERS



# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TRAIN_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH
VAL_SET_IMAGES_PATH = OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH
TEST_SET_IMAGES_PATH = OCCLUDED_TEST_SET_IMAGES_PATH

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
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
parser.add_argument('--freezing_module', type=int, required=True, choices=FREEZING_MODULES_LAYERS.keys(), help=f"Freezing module. Choose from: {list(FREEZING_MODULES_LAYERS.keys())}")
parser.add_argument('--patience', type=int, required=False, default=3, help='Early stopping patience (default: 3)')
parser.add_argument('--redirect_output', action='store_true', help='If set, redirect stdout and stderr to a log file')
parser.add_argument('--quick', action='store_true', help='If set, use a smaller subset of the data for a quick test run')
args = parser.parse_args()

if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__{__name__}__console_output.txt")

    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 
else: 
    LOG_FILE_PATH = None

if os.name == 'nt':  # Windows
    tracking_uri = None
    experiment_name = f"try_training_{args.model_name}"
    mlflow.set_experiment(experiment_name)
else:
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
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training_yolo.py" --model_name yolo_last --batch_size 64 --epochs 10 --learning_rate 0.0001 --dropout_rate 0.3 --freezing_module 2 --quick
# >>> YOLO local test
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training_yolo.py" --model_name yolo_last --batch_size 8  --epochs 10 --learning_rate 0.0001 --dropout_rate 0.3 --freezing_module 2 --quick --redirect_output

if __name__ == "__main__":
    training_folder_path = TRAIN_SET_IMAGES_PATH
    val_folder_path = VAL_SET_IMAGES_PATH
    test_folder_path = TEST_SET_IMAGES_PATH

    model = load_yolo_model(args.model_name)

    with mlflow.start_run():
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        train_model_yolo_training_run(model, training_folder_path, val_folder_path,
                                      epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                      freezing_module=args.freezing_module, dropout_rate=args.dropout_rate, patience=args.patience,
                                      quick_run=args.quick)

        test_acc = evaluate_model_yolo_training_run(model, test_folder_path, debug=False)

        save_model_yolo_training_run(model, args.model_name, test_acc)
