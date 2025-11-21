import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.config import ADELE_TEST_SET_H5_PATH
from modules.data import load_test_generator
from modules.landmark_utils import detect_facial_landmarks
import matplotlib.pyplot as plt

# Function to plot image with landmarks
def plot_image_with_landmarks(image, landmarks):
    plt.imshow(image)
    for (x, y) in landmarks:
        plt.scatter(x, y, c='red', s=10)  # Plot landmarks as red dots
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Before knowing what type of image ill have to see if it works, I need to implement the pipeline even if i dont have the occlusion part yet
    test_generator = load_test_generator(ADELE_TEST_SET_H5_PATH, 'test')

    for batch in test_generator:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch = batch[0], batch[1]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
        
        for image in X_batch:
            landmarks = detect_facial_landmarks(image)
            print(f"Detected landmarks: {landmarks}")
            plot_image_with_landmarks(image, landmarks)  # Plot the image with landmarks

