import os; import sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mlflow
import numpy as np
import argparse
from sklearn.model_selection import ParameterGrid

from modules.config import MLFLOW_DIR, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.data__load import load_offline_data_generators
from modules.data import refresh_show_flags
from modules.model import build_model_occfinetuning
from modules.train_eval_save import addestra_modello, valuta_modello
from modules.misc import Tee, get_timestamp



ORIGINAL_TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
OCCLUDED_TRAINVAL_SET_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH

LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__{__name__}__console_output.txt")

SAVE_ALL_MODELS = False
SHORT_TRAINING_FOR_TESTING = False



def trova_num_layers(initial_bias, model_name):
    model = build_model_occfinetuning(1e-4, 0.3, 1e-3, initial_bias, model_name, unfreeze=0)  # unfreeze=0 means all layers frozen, we just want to count them here
    return len(model.layers)

# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/train_eval/do_find_unfreeze.py" --model_name PattLite --learning_rate 1e-4 --l2_reg 0.002
def main():
    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Testing different layers accuracy for Final Layers')
    parser.add_argument('--model_name',     type=str, required=True,    help='Model name. Default is PattLite', default='PattLite')
    parser.add_argument('--batch_size',     type=int, default=64,       help='Batch size for data generators. Default is 64')
    parser.add_argument('--learning_rate',  type=float, required=True,  help='Learning rate. Default is 1e-4', default=1e-4)
    parser.add_argument('--l2_reg',         type=float, required=True,  help='L2 regularization. Default is 1e-3', default=1e-3)
    parser.add_argument('--debug_prints',          action='store_true', help='If set, print debug information during training')
    parser.add_argument('--redirect_output', action='store_true', help='If set, redirect stdout and stderr to a log file, apart from showing it to console.')
    # parser.add_argument('--short_training_for_testing', action='store_true', help='If set, use a very short training (e.g. 1 epoch or 100 steps) just to test that the training loop works without waiting for a whole epoch to end.')
    args = parser.parse_args()

    # if args.short_training_for_testing:
    if SHORT_TRAINING_FOR_TESTING:
        GLOBALS["SHORT_TRAINING_FOR_TESTING"] = True
        refresh_show_flags()
        N_ITER_SEARCH = 2
    else:
        N_ITER_SEARCH = 10  

    if args.redirect_output:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = Tee(LOG_FILE_PATH)
        sys.stderr = Tee(LOG_FILE_PATH) 

    TRAIN_EPOCH = 10
    TRAIN_ES_PATIENCE = 3
    TRAIN_LR_PATIENCE = 2
    ES_LR_MIN_DELTA = 0.0001
    TRAIN_MIN_LR = 1e-6
    model_name = args.model_name

    os.makedirs(MLFLOW_DIR, exist_ok=True)
    tracking_uri = f"file://{os.path.abspath(MLFLOW_DIR)}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"find_unfreeze_{model_name}"
    mlflow.set_experiment(experiment_name)

    print(f"=============================== SETTINGS ==================================")
    print(f"CONSTANTS: ")
    print(f"\tORIGINAL_TRAINVAL_SET_PATH: {ORIGINAL_TRAINVAL_SET_PATH}")
    print(f"\tOCCLUDED_TRAINVAL_SET_PATH: {OCCLUDED_TRAINVAL_SET_PATH}")
    print(f"\tTEST_SET_PATH: {TEST_SET_PATH}")
    print(f"ARGS: ")
    print(f"\tmodel_name: {model_name}")
    print(f"\tbatch_size: {args.batch_size}")
    print(f"\tlearning_rate: {args.learning_rate}")
    print(f"\tl2_reg: {args.l2_reg}")
    print(f"\tdebug_prints: {args.debug_prints}")
    print(f"\tredirect_output: {args.redirect_output}")
    # print(f"\tshort_training_for_testing: {args.short_training_for_testing}")
    print(f"TRAINING PARAMS:")
    print(f"\tTRAIN_EPOCH: {TRAIN_EPOCH}")
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

    # Carica i dati
    train_generator, valid_generator, test_generator, initial_bias = load_offline_data_generators(
                                                            # Paths ---------------------------------------------------
                                                            original_trainval_path=ORIGINAL_TRAINVAL_SET_PATH,
                                                            occluded_trainval_path=OCCLUDED_TRAINVAL_SET_PATH,
                                                            occluded_test_path=TEST_SET_PATH,
                                                            # Occlusion parameters ------------------------------------

                                                            # Command line args for working ---------------------------
                                                            batch_size=args.batch_size,
                                                        ) 

    
    total_layers = trova_num_layers(initial_bias,model_name)
    print(f"Total layers in the model: {total_layers}")
    
    num_layers_to_unfreeze = list(range(1, total_layers + 1))
    # Definisci una griglia di parametri per il numero di layer da scongelare
    param_dist = {
                  'learning_rate': [1e-5, args.learning_rate],
                  'dropout_rate': [0.3, 0.5],
                  }

    with mlflow.start_run():  # parent run for the whole experiment
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("l2_reg", args.l2_reg)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("train_epochs", TRAIN_EPOCH)
        mlflow.log_param("train_es_patience", TRAIN_ES_PATIENCE)
        mlflow.log_param("train_lr_patience", TRAIN_LR_PATIENCE)
        mlflow.log_param("es_lr_min_delta", ES_LR_MIN_DELTA)
        mlflow.log_param("train_min_lr", TRAIN_MIN_LR)

        overall_best_accuracy = 0
        overall_best_model = None
        overall_best_config = None

        for params in ParameterGrid(param_dist):
            best_accuracy = 0
            best_params = None
            n_iter_search = N_ITER_SEARCH
            chosen_numbers = set()
            for trial_idx in range(n_iter_search):
                valid_choices = [x for x in num_layers_to_unfreeze if x != 0]
                unfreeze = np.random.choice([x for x in valid_choices if x not in chosen_numbers])
                chosen_numbers.add(unfreeze)

                # start a nested run for this single trial (params + unfreeze)
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_param("unfreeze", int(unfreeze))
                    mlflow.log_param("trial_idx", int(trial_idx))

                    if args.debug_prints:
                        print(f"Trial {trial_idx+1}/{n_iter_search} with params:")
                        print(f"\tnum_layers_to_unfreeze: {unfreeze}")
                        print(f"\tlearning_rate: {params['learning_rate']}")
                        print(f"\tdropout_rate: {params['dropout_rate']}")

                    model = build_model_occfinetuning(params['learning_rate'], params['dropout_rate'], args.l2_reg, initial_bias, model_name, unfreeze=unfreeze)
                    history = addestra_modello(model, train_generator, valid_generator, test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, None, model_name)

                    test_loss, test_acc = valuta_modello(model, test_generator, None, model_name)

                    # compute and log final trial metrics (val best / loss at best)
                    val_acc = float(np.max(history.history['val_categorical_accuracy']))
                    best_epoch = int(np.argmax(history.history['val_categorical_accuracy']))
                    val_loss_at_best = float(history.history['val_loss'][best_epoch])
                    mlflow.log_metric("val_acc_best", val_acc)
                    mlflow.log_metric("val_loss_at_best", val_loss_at_best)
                    mlflow.log_metric("test_acc", test_acc)
                    mlflow.log_metric("test_loss", test_loss)

                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_params = {
                            'num_layers_to_unfreeze': unfreeze,
                            'learning_rate': params['learning_rate'],
                            'dropout_rate': params['dropout_rate']
                        }

                        mlflow.log_metric("best_accuracy_so_far", best_accuracy)


                        if SAVE_ALL_MODELS:
                            # Save the model for this trial
                            mlflow.keras.log_model(
                                model,
                                artifact_path="best_model",
                                registered_model_name=None  # optional: set if using MLflow Model Registry
                            )
                    
                    if val_acc > overall_best_accuracy:
                        overall_best_accuracy = val_acc
                        overall_best_model = model
                        overall_best_config = {
                            'num_layers_to_unfreeze': unfreeze,
                            'learning_rate': params['learning_rate'],
                            'dropout_rate': params['dropout_rate']
                        }

                    

        # Save the overall best model and its config
        mlflow.log_metric("final_best_accuracy", overall_best_accuracy)
        mlflow.log_param("overall_best_unfreeze", overall_best_config['num_layers_to_unfreeze'])
        mlflow.log_param("overall_best_learning_rate", overall_best_config['learning_rate'])
        mlflow.log_param("overall_best_dropout_rate", overall_best_config['dropout_rate'])
        mlflow.keras.log_model(
            overall_best_model,
            artifact_path="overall_best_model"
        )


if __name__ == "__main__":
    main()