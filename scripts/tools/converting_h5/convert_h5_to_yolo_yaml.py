import os
import h5py
import numpy as np
from PIL import Image
import yaml

from modules.config import DATASETS_PATH

def convert_h5_to_yolo(h5_path, output_dir, debug=True):
    """
    Convert an H5 test set to YOLO-compatible format.

    Args:
        h5_path (str): Path to the H5 file.
        output_dir (str): Directory to save YOLO-formatted data.

    Returns:
        None
    """
    # Create YOLO directory structure
    images_dir = os.path.join(output_dir, "images/test")
    labels_dir = os.path.join(output_dir, "labels/test")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Read data from H5 file
    with h5py.File(h5_path, "r") as f:
        X_test = np.array(f["X_test"])  # Shape: (num_samples, height, width, channels)
        y_test = np.array(f["y_test"])  # Shape: (num_samples,)
        class_names = [name.decode('utf-8') for name in f["class_names"][...]]  # Decode bytes to strings

        if debug:
            print(f"Class names from H5: {class_names}")

        # Optional: Read paths if available
        paths_data = f["paths"][...] if "paths" in f else None

    # Save images and labels
    for i, (image, label) in enumerate(zip(X_test, y_test)):
        # Save image
        image = Image.fromarray((image * 255).astype(np.uint8)) if image.max() <= 1 else Image.fromarray(image)
        image_path = os.path.join(images_dir, f"{i:06d}.jpg")
        image.save(image_path)

        # Save label
        label_path = os.path.join(labels_dir, f"{i:06d}.txt")
        with open(label_path, "w") as label_file:
            label_file.write(f"{label} 0.5 0.5 1.0 1.0\n")  # YOLO format: class_id x_center y_center width height

    # Generate data.yaml
    data_yaml = {
        "path": output_dir,
        "train": "images/train",  # Placeholder
        "val": "images/val",      # Placeholder
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data_yaml, yaml_file)

    print(f"Conversion complete! YOLO data saved to: {output_dir}")

# DATASET = "adele_test_set"
DATASET = "occluded_test_set"

if __name__ == "__main__":
    h5_path = os.path.join(DATASETS_PATH, DATASET, f"{DATASET}.h5")
    yaml_folder_path = os.path.join(DATASETS_PATH, DATASET, f"{DATASET}_yaml")

    convert_h5_to_yolo(h5_path, yaml_folder_path)