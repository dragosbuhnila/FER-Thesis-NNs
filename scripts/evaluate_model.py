import os
import sys
import time
import tensorflow as tf

from modules.config import ACCURACY_RESULTS_PATH, ADELE_TEST_SET_H5_PATH, OCCLUDED_TEST_SET_PATH, OCCLUDED_TEST_SET_RESIZED_PATH, OCCLUDED_TEST_SET_H5_PATH, ALL_MODELS_PATHS
from modules.data import generate_h5_from_images, load_data_generator
from modules.model import load_model


PATHS = {
    "ADELE": {
        "test_set": None,
        "test_set_resized": None,
        "test_set_h5": ADELE_TEST_SET_H5_PATH
    },
    "OCCLUDED": {
        "test_set": OCCLUDED_TEST_SET_PATH,
        "test_set_resized": OCCLUDED_TEST_SET_RESIZED_PATH,
        "test_set_h5": OCCLUDED_TEST_SET_H5_PATH
    }
}


# ============== MACROS ===============
MODEL_PATHS_SUBSET = ALL_MODELS_PATHS
TEST_SET = "ADELE"  # Options: "ADELE", "OCCLUDED"
# MODELS_NAMES = ["resnet_finetuning", "pattlite_finetuning", "vgg19_finetuning", "inceptionv3_finetuning", "convnext_finetuning", "efficientnet_finetuning"]
MODELS_NAMES = ["efficientnet_finetuning"]

REDIRECT_OUTPUT = False
LOG_FILE = os.path.join(ACCURACY_RESULTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}_accuracies_{TEST_SET.lower()}.log")
# =========== END OF MACROS ===========


if REDIRECT_OUTPUT:
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f" - {gpu}")
    # Set memory growth to avoid allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. The code will run on the CPU.")



def evaluate_model(model, model_name, test_generator):
    if "yolo" in model_name:
        return None
    
    # For other models. Probably: resnet, vgg, inception, convnext, pattlite, efficientnet
    test_loss, test_acc = model.evaluate(test_generator)
    return test_loss, test_acc

if __name__ == "__main__":
    # 1) Load the test set
    # if you can't find the h5 file, generate it from the images
    if not os.path.exists(PATHS[TEST_SET]["test_set_h5"]):
        generate_h5_from_images(PATHS[TEST_SET]["test_set"], PATHS[TEST_SET]["test_set_resized"], PATHS[TEST_SET]["test_set_h5"])
    test_generator = load_data_generator(PATHS[TEST_SET]["test_set_h5"], 'test')

    # 2) Run the evaluations on the test set
    models_results = {name: {"test_loss": None, "test_acc": None} for name in MODELS_NAMES}

    for model_name in MODELS_NAMES:
        print("======================================")
        print(f"Evaluating model: {model_name}")

        # a) Load the model
        model = load_model(model_name, MODEL_PATHS_SUBSET)
        if model is None:
            print("Model loading not implemented for this model type.")
            continue
        else:
            # b) Evaluate the model
            test_loss, test_acc = evaluate_model(model, model_name, test_generator)
            models_results[model_name]["test_loss"] = test_loss
            models_results[model_name]["test_acc"] = test_acc
    print("======================================")

    # 3) Print the final results
    print(f"\n\nFinal evaluation results on {TEST_SET.lower()} test set:")
    for model_name, results in models_results.items():
        print(f"Model: {model_name} - Test Loss: {results['test_loss']:.4f}, Test Accuracy: {results['test_acc']:.4f}")
    