import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras

from modules.data__load import load_offline_data_generators
from modules.data import refresh_show_flags
from modules.model import build_model_occfinetuning
from modules.config import ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, OCCLUDED_TEST_SET_H5_PATH, MLFLOW_DIR, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.train_eval import addestra_modello, salva_modello, valuta_modello; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from modules.misc import Tee, get_timestamp



# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
ORIGINAL_TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
OCCLUDED_TRAINVAL_SET_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH

LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__{__name__}__console_output.txt")



# example usage: 
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training.py" --batch_size 8 --matching_amount 0.2 --val_occ_prob 0.5 --occ_prob 0.2 --model_name ConvNeXt --FT_EPOCH 300  --dropout_rate 0.5 --learning_rate  0.0001 --l2_reg 0.00200041712955451
def main():
    # *init Neptune or alternative*

    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Training parameters for occlusion finetuning')
    parser.add_argument('--unfreeze', type=int, required=True, help='Number of layers to unfreeze for fine-tuning. If not provided.')
    parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization parameter')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
    parser.add_argument('--FT_EPOCH', type=int, required=True, help='Training epochs')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=64)
    parser.add_argument('--redirect_output', action='store_true', help='If set, redirect stdout and stderr to a log file')
    parser.add_argument('--gen_train_occlusion_ratio', type=float, required=False, help='Occlusion ratio for generating occluded training samples. Default is 0.8', default=0.8)
    parser.add_argument('--long_epochs', action='store_true', help='If set, epochs len will be like len(occluded_images) instead of len(original_images). The former is 250k+ images, the latter is 20k+ images. ')
    args = parser.parse_args()

    # Recupera i parametri dalla linea di comando
    unfreeze = args.unfreeze
    l2_reg = args.l2_reg
    FT_LR = args.learning_rate
    FT_DROPOUT = args.dropout_rate
    FT_EPOCH = args.FT_EPOCH
    model_name = args.model_name
    long_epochs = args.long_epochs
    batch_size = args.batch_size
    gen_train_occlusion_ratio = args.gen_train_occlusion_ratio

    if long_epochs:
        GLOBALS["LONG_EPOCHS"] = True
        refresh_show_flags()

    # others
    TRAIN_ES_PATIENCE = 3
    TRAIN_LR_PATIENCE = 2
    ES_LR_MIN_DELTA = 0.0001
    TRAIN_MIN_LR = 1e-6

    if args.redirect_output:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = Tee(LOG_FILE_PATH)
        sys.stderr = Tee(LOG_FILE_PATH) 

    tracking_uri = f"file://{os.path.abspath(MLFLOW_DIR)}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"try_training_{model_name}"
    mlflow.set_experiment(experiment_name)

    print(f"================================ SETTINGS ==================================")
    print(f"CONSTANTS: ")
    print(f"\tORIGINAL_TRAINVAL_SET_PATH: {ORIGINAL_TRAINVAL_SET_PATH}")
    print(f"\tOCCLUDED_TRAINVAL_SET_PATH: {OCCLUDED_TRAINVAL_SET_PATH}")
    print(f"\tTEST_SET_PATH: {TEST_SET_PATH}")
    print(f"ARGS: ")
    print(f"\tunfreeze: {unfreeze}")
    print(f"\tl2_reg: {l2_reg}")
    print(f"\tlearning_rate: {FT_LR}")
    print(f"\tdropout_rate: {FT_DROPOUT}")
    print(f"\ttraining_epochs: {FT_EPOCH}")
    print(f"\tmodel_name: {model_name}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\tlong_epochs: {long_epochs}")
    print(f"\tgen_train_occlusion_ratio: {gen_train_occlusion_ratio}")
    print(f"TRAINING PARAMS:")
    print(f"\tTRAIN_ES_PATIENCE: {TRAIN_ES_PATIENCE}")
    print(f"\tTRAIN_LR_PATIENCE: {TRAIN_LR_PATIENCE}")
    print(f"\tES_LR_MIN_DELTA: {ES_LR_MIN_DELTA}")
    print(f"\tTRAIN_MIN_LR: {TRAIN_MIN_LR}")
    print(f"MLFLOW:")
    print(f"\ttracking_uri: {tracking_uri}")
    print(f"\texperiment_name: {experiment_name}")
    print(f"GLOBALS:")
    for key, value in GLOBALS.items():
        print(f"\t{key}: {value}")
    print(f"===========================================================================")


    train_generator, valid_generator, test_generator, initial_bias = load_offline_data_generators(
                                                                                                    # Paths ---------------------------------------------------
                                                                                                    original_trainval_path=ORIGINAL_TRAINVAL_SET_PATH,
                                                                                                    occluded_trainval_path=OCCLUDED_TRAINVAL_SET_PATH,
                                                                                                    occluded_test_path=TEST_SET_PATH,
                                                                                                    # Occlusion parameters ------------------------------------

                                                                                                    # Command line args for working ---------------------------
                                                                                                    batch_size=batch_size,
                                                                                                    training_occlusion_probability=gen_train_occlusion_ratio
                                                                                                ) 

    model = build_model_occfinetuning(FT_LR, FT_DROPOUT, l2_reg, initial_bias, model_name, unfreeze=unfreeze)

    # # Logga i parametri di addestramento su Neptune
    # run[f"{model_name}finetuning/parameters"] = {
    #     "learning_rate": FT_LR,
    #     "dropout_rate": FT_DROPOUT,
    #     "l2_reg": l2_reg,
    #     "epochs": FT_EPOCH,
    #     "batch_size": batch_size
    # }
    run = None


    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("learning_rate", FT_LR)
        mlflow.log_param("dropout_rate", FT_DROPOUT)
        mlflow.log_param("l2_reg", l2_reg)
        mlflow.log_param("epochs", FT_EPOCH)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("unfreeze", unfreeze)
        mlflow.log_param("gen_train_occlusion_ratio", gen_train_occlusion_ratio)

        history = addestra_modello(model, train_generator, valid_generator, test_generator, FT_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, run, model_name)
        test_loss, test_acc = valuta_modello(model, test_generator, run, model_name)
        salva_modello(model, run, model_name, test_acc)

    ## *** termina Neptune run o alternativa ***

if __name__ == "__main__":
    main()