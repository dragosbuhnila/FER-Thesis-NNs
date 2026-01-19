import os
from PIL import Image
import h5py
import numpy as np
import hashlib

def hash_image(image):
    # If a PIL Image, convert to numpy in a deterministic way
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    arr = np.ascontiguousarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()

def print_npy(npy_file_path, output_file_path):
    data = np.load(npy_file_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()  # Convert 0-dim array to its content

    # Print the contents
    with open(output_file_path, 'w') as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"Index: {key} -> Hash: {value}\n")
        elif isinstance(data, list):
            for index, value in enumerate(data):
                f.write(f"Index: {index} -> Value: {value}\n")
        else:
            f.write(f"Contents of {npy_file_path}:\n")
            f.write(data.__str__())

def generate_h5_from_images(test_set_path, resized_path, h5_path):
    class_names = sorted(os.listdir(test_set_path))
    paths = []
    X_test = []
    y_test = []
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(test_set_path, class_name)
        image_files = sorted(os.listdir(class_folder))
        for image_file in image_files:
            image_path = os.path.join(class_folder, image_file)
            # Load PNG image as RGB NumPy array and resize to (128, 128, 3)
            image = np.array(Image.open(image_path).convert('RGB').resize((128, 128)))
            X_test.append(image)
            y_test.append(class_idx)  # Store class index instead of name
            paths.append(image_path)

            # Also save the images to resized_path
            save_folder = os.path.join(resized_path, class_name)
            os.makedirs(save_folder, exist_ok=True)
            Image.fromarray(image).save(os.path.join(save_folder, image_file))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 3) Save new h5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)  # Now integers
        f.create_dataset("class_names", data=np.array(class_names).astype('S'))  # Save as bytes
        f.create_dataset("paths", data=np.array(paths).astype('S'))  # Save as bytes
    print(f"Saved {X_test.shape[0]} images to {h5_path}")

    # 4) Check saved h5 to have 350 images, 7 classes, 50 images per class
    with h5py.File(h5_path, "r") as f:
        if "X_test" not in f.keys():
            raise ValueError("X_test not found in the H5 file.")
        if "y_test" not in f.keys():
            raise ValueError("y_test not found in the H5 file.")
        if "class_names" not in f.keys():
            raise ValueError("class_names not found in the H5 file.")
        if "paths" not in f.keys():
            raise ValueError("paths not found in the H5 file.")
        
        X_test_loaded = np.array(f["X_test"])
        y_test_loaded = np.array(f["y_test"])
        class_names_loaded = [name.decode('utf-8') for name in f["class_names"][...]]
        paths_loaded = [path.decode('utf-8') for path in f["paths"][...]]

        if X_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 images, but found {X_test_loaded.shape[0]}.")
        if y_test_loaded.shape[0] != 350:
            raise ValueError(f"Expected 350 labels, but found {y_test_loaded.shape[0]}.")
        if len(class_names_loaded) != 7:
            raise ValueError(f"Expected 7 classes, but found {len(class_names_loaded)}.")
        if len(paths_loaded) != 350:
            raise ValueError(f"Expected 350 paths, but found {len(paths_loaded)}.")
