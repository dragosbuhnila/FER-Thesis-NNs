from PIL import Image
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