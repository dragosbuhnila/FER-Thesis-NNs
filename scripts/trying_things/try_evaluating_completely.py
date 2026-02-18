import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.model import load_model
from modules.visualize import plot_image;
from modules.data__load import load_test_generator
from modules.evaluate import evaluate_keras_model
from modules.config import OCCLUDED_TEST_SET_H5_PATH



MODEL_NAME = "occft_convnext"
TEST_SET = OCCLUDED_TEST_SET_H5_PATH



if __name__ == "__main__":
    print("Loading test generator...")
    test_generator = load_test_generator(TEST_SET)  # Set to True for quick testing, False for full evaluation

    print("Test generator loaded.")

    # for batch in test_generator:
    #     for image, label_probs in zip(batch[0], batch[1]):
    #         label = label_probs.argmax()
    #         plot_image(image, f"True Label: {EMOTIONS[label]} (idx={label})")

    model = load_model(MODEL_NAME)
    accuracies, precision_recall_f1, probabilities, y_true, y_pred = evaluate_keras_model(model, test_generator, MODEL_NAME)

    print(f"=======================================================================================")
    print(f"exited modules.evaluate")
    print(f"=======================================================================================")
    print("Accuracies:", accuracies)
    print("Precision, Recall, F1-score:", precision_recall_f1)
    print("Probabilities shape:", probabilities.shape)
    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)
    