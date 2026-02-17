import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.model import load_model
from modules.visualize import plot_image;
from modules.data__load import load_test_generator
from modules.evaluate import evaluate_keras_model
from modules.config import OCCLUDED_TEST_SET_H5_PATH



if __name__ == "__main__":
    print("Loading test generator...")
    test_generator = load_test_generator(OCCLUDED_TEST_SET_H5_PATH)
    print("Test generator loaded.")

    # for batch in test_generator:
    #     for image, label_probs in zip(batch[0], batch[1]):
    #         label = label_probs.argmax()
    #         plot_image(image, f"True Label: {EMOTIONS[label]} (idx={label})")

    model = load_model("occft_convnext")
    probabilities, y_true, y_pred = evaluate_keras_model(model, test_generator, "ConvNeXtOccFT")
    