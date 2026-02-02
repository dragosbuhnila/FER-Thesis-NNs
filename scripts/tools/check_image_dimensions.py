import numpy as np
from PIL import Image

def check_image_dimensions(image_path):
    """
    Check and print the dimensions of an image as a NumPy array.
    """
    try:
        # Open the image and convert it to a NumPy array
        with Image.open(image_path) as img:
            img_array = np.array(img)
            print(f"Image dimensions (height, width, channels): {img_array.shape}")
            print(f"Data Type: {img_array.dtype}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = "C:\\Users\\Dragos\\Roba\\Lectures\\YM2.2\\Thesis\\e Models\\data\\datasets\\occluded_train_val_set\\images\\train\\gt-angry_occ-fear\\00cd28ba22c246af733c4e0d8c6da551_gt-angry_occ-fear_mismatching_negative.png"  # Replace with your image path
check_image_dimensions(image_path)