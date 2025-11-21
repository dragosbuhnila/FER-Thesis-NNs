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