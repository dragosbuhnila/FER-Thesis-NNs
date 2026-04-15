import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from modules.xai_bubbles import get_masks

# load an example image file (or replace with any numpy image array)
img_path = r"C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\data\datasets\adele_test_set\extracted_images\ANGRY\image_3_ANGRY_3a30de5b8a5254fa675cc0192f56d06a.png"
img = Image.open(img_path).convert("RGB")
img_array = np.array(img)

# get masked images and masks
masked_images, masks = get_masks(img_array, bubble_radius=26, num_bubbles=10, iterations=64)

# convert masked images (float in [0,1]) to uint8
perturbed = [(p * 255).astype(np.uint8) for p in masked_images]

# prepare images: put original first, then all perturbed
all_images = [img_array.astype(np.uint8)] + perturbed

images_per_page = 3 * 6  # 3 cols x 6 rows = 18 images per figure
for page_start in range(0, len(all_images), images_per_page):
    chunk = all_images[page_start:page_start + images_per_page]
    n = len(chunk)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    for idx in range(rows * cols):
        ax = axs[idx]
        if idx < n:
            ax.imshow(chunk[idx])
            title = "Original" if (page_start == 0 and idx == 0) else f"Perturbed #{page_start + idx}"
            ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Perturbations (images {page_start + 1}–{page_start + n})")
    plt.tight_layout()
    plt.show()