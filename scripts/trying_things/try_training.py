import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras

from modules.data__load import load_data_generators
from modules.model import build_model_occfinetuning
from modules.config import ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
from modules.train_eval import addestra_modello, salva_modello, valuta_modello; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = ADELE_TEST_SET_H5_PATH  
TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH



# example usage: 
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/try_training.py" --batch_size 8 --matching_amount 0.2 --val_occ_prob 0.5 --occ_prob 0.2 --model_name ConvNeXt --FT_EPOCH 300  --dropout_rate 0.5 --learning_rate  0.0001 --l2_reg 0.00200041712955451
def main():
    # *init Neptune or alternative*

    # Definisci gli argomenti della linea di comando
    parser = argparse.ArgumentParser(description='Training parameters for occlusion finetuning')
    parser.add_argument('--l2_reg', type=float, required=True, help='L2 regularization parameter')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate')
    parser.add_argument('--FT_EPOCH', type=int, required=True, help='Training epochs')
    parser.add_argument('--model_name', type=str, required=True, help='Model name. Default is PattLite', default='PattLite')
    parser.add_argument('--occ_prob', type=float, required=True, help='Occlusion probability')
    parser.add_argument('--val_occ_prob', type=float, required=False, help='Validation occlusion probability', default=0.5)
    parser.add_argument('--matching_amount', type=float, required=False, help='Amount of matching for occlusions (float). Exaple: 0.2 is 20%, i.e. out of 50 images 10 will be matching, the rest will be of some mismatch type (every 4)', default=0.2)
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=64)
    args = parser.parse_args()

    # Recupera i parametri dalla linea di comando
    l2_reg = args.l2_reg
    FT_LR = args.learning_rate
    FT_DROPOUT = args.dropout_rate
    FT_EPOCH = args.FT_EPOCH
    model_name = args.model_name
    occlusion_probability = args.occ_prob
    val_occlusion_probability = args.val_occ_prob
    matching_amount = args.matching_amount
    batch_size = args.batch_size

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"try_training_{model_name}")

    # Carica i dati
    train_generator, valid_generator, test_generator, initial_bias = load_data_generators(TRAINVAL_SET_PATH, TEST_SET_PATH, 
                                                                                          occlusion_probability=occlusion_probability, 
                                                                                          masking_function="lines", 
                                                                                          use_label_smoothing=True, 
                                                                                          mismatch=True,
                                                                                          small_subset=False, 
                                                                                          matching_amount=matching_amount, # same matching amount as the test set, where only one in 5 images is matching positive
                                                                                          batch_size=batch_size,
                                                                                          validation_occlusion_probability=val_occlusion_probability,
                                                                                          ) 

    model = build_model_occfinetuning(FT_LR, FT_DROPOUT, l2_reg, initial_bias, model_name)

    # # Logga i parametri di addestramento su Neptune
    # run[f"{model_name}finetuning/parameters"] = {
    #     "learning_rate": FT_LR,
    #     "dropout_rate": FT_DROPOUT,
    #     "l2_reg": l2_reg,
    #     "epochs": FT_EPOCH,
    #     "batch_size": 64
    # }
    run = None

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("learning_rate", FT_LR)
        mlflow.log_param("dropout_rate", FT_DROPOUT)
        mlflow.log_param("l2_reg", l2_reg)
        mlflow.log_param("epochs", FT_EPOCH)
        mlflow.log_param("batch_size", 64)

        mlflow.log_param("occlusion_probability", occlusion_probability)
        mlflow.log_param("val_occlusion_probability", val_occlusion_probability)
        mlflow.log_param("matching_amount", matching_amount)

        
        history = addestra_modello(model, train_generator, valid_generator, test_generator, FT_EPOCH, 50, 15, 0.003, 1e-6, run, model_name)

        # Valuta il modello
        _, _ = valuta_modello(model, test_generator, run, model_name)

        # Salva il modello e la storia dell'addestramento
        salva_modello(model, run, model_name)


    ## *** termina Neptune run o alternativa ***

if __name__ == "__main__":
    main()