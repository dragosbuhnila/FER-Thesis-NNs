import os
import mlflow
import numpy as np
import argparse
from sklearn.model_selection import ParameterGrid

from modules.config import MLFLOW_DIR, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
from modules.data__load import load_offline_data_generators
from modules.model import build_model_occfinetuning
from modules.train_eval import addestra_modello



ORIGINAL_TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
OCCLUDED_TRAINVAL_SET_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH



def trova_num_layers(initial_bias, model_name):
    model = build_model_occfinetuning(1e-4, 0.3, 1e-3, initial_bias, model_name)
    return len(model.layers)

# & ... --model_name PattLite --learning_rate 1e-4 --l2_reg 0.002
def main():
    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Testing different layers accuracy for Final Layers')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data generators. Default is 64')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate. Default is 1e-4', default=1e-4)
    parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization. Default is 1e-3', default=1e-3)
    args = parser.parse_args()


    TRAIN_EPOCH = 10
    TRAIN_ES_PATIENCE = 3
    TRAIN_LR_PATIENCE = 2
    ES_LR_MIN_DELTA = 0.0001
    TRAIN_MIN_LR = 1e-6
    model_name = args.model_name

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
    print(f"TRAINING PARAMS:")
    print(f"\tTRAIN_EPOCH: {TRAIN_EPOCH}")
    print(f"\tTRAIN_ES_PATIENCE: {TRAIN_ES_PATIENCE}")
    print(f"\tTRAIN_LR_PATIENCE: {TRAIN_LR_PATIENCE}")
    print(f"\tES_LR_MIN_DELTA: {ES_LR_MIN_DELTA}")
    print(f"\tTRAIN_MIN_LR: {TRAIN_MIN_LR}")
    print(f"MLFLOW:")
    print(f"\ttracking_uri: {tracking_uri}")
    print(f"\texperiment_name: {experiment_name}")
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
    
    num_layers_to_unfreeze = list(range(1, total_layers + 1))
    # Definisci una griglia di parametri per il numero di layer da scongelare
    param_dist = {
                  'learning_rate': [1e-5, args.learning_rate],
                  'dropout_rate': [0.3, 0.5],
                  }

    for params in ParameterGrid(param_dist):
        # Esegui la ricerca randomica
        best_accuracy = 0
        best_params = None
        n_iter_search = 10  # Numero di iterazioni di ricerca randomica
        chosen_numbers = set()
        for _ in range(n_iter_search):
            # Filtra la lista dei valori possibili per escludere 0
            valid_choices = [x for x in num_layers_to_unfreeze if x != 0]

            # Scegli un numero di layer da scongelare casualmente dalla lista filtrata, assicurandoti che non sia già stato scelto
            unfreeze = np.random.choice([x for x in valid_choices if x not in chosen_numbers])
            chosen_numbers.add(unfreeze)
            
            # Ricostruisci il modello con il numero specificato di layer da scongelare
            model = build_model_occfinetuning(params['learning_rate'], params['dropout_rate'], args.l2_reg, initial_bias, model_name, unfreeze=unfreeze)
            
            # Addestra il modello
            history = addestra_modello(model, train_generator, valid_generator, test_generator, TRAIN_EPOCH, TRAIN_ES_PATIENCE, TRAIN_LR_PATIENCE, ES_LR_MIN_DELTA, TRAIN_MIN_LR, None, model_name)
            
            val_acc = max(history.history['val_categorical_accuracy'])
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = {'num_layers_to_unfreeze': unfreeze, 'learning_rate': params['learning_rate'], 'dropout_rate': params['dropout_rate']}
                # run[f"{model_name}/finetuning"].append(f"accuracy= {best_accuracy} with {best_params}")
                mlflow.log_metric("best_accuracy", best_accuracy)

        print(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
        # run[f"{model_name}/finetuning"].append(f"FINISHED...")
        # run[f"{model_name}/finetuning"].append(f"Best parameters: {best_params} with accuracy: {best_accuracy}")
        mlflow.log_metric("final_best_accuracy", best_accuracy)

if __name__ == "__main__":
    main()