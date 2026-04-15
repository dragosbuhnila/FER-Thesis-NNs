# visualize_perturbations.py
import math
import sys, os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_black_square_images(img_array, square_sizes):
    """
    Creates images with a black square placed only if the center of the square is over a non-zero pixel.
    Skips generating images in the 8-connected pixels around any center where a square has already been generated.

    :param image_path: Path to the input image.
    :param output_folder: Folder to save the output images.
    :param square_size: Size of the black square to be placed on the image.
    """

    for square_size in square_sizes:
        # Ensure the square size is odd
        if square_size % 2 == 0:
            square_size += 1  # Make it odd by adding 1

        neighbors = square_size // 5  # Number of neighbors to check

        height, width, channels = img_array.shape

        # Offset for the center of the square
        offset = square_size // 2

        # Create a boolean mask to track visited centers
        visited = np.zeros((height, width), dtype=bool)

        total_positions = (height - square_size + 1) * (width - square_size + 1)

        masked_images = []

        with tqdm(total=total_positions, desc=f"Generating images with black squares of size {square_size}") as pbar:
            counter = 0
            for y in range(offset, height - offset):
                for x in range(offset, width - offset):
                    # Skip this center if it or any of its 8-connected neighbors have already been visited
                    if visited[y, x]:
                        pbar.update(1)
                        continue

                    # Check if the center pixel of the current square is non-zero
                    if np.any(img_array[y, x] != 0):
                        # Create a copy of the original image
                        masked_image = img_array.copy()

                        # Apply the black square by setting pixels in the square to 0
                        masked_image[y-offset:y+offset+1, x-offset:x+offset+1] = 0  # Black square (RGB [0, 0, 0])

                        # Convert the image array to uint8 if it's not already
                        masked_image = masked_image.astype(np.uint8)

                        # Convert the NumPy array to a PIL Image object
                        masked_image = Image.fromarray(masked_image)

                        # Save the image as an RGB PNG file
                        masked_images.append(np.array(masked_image))
                        counter += 1

                        # Mark the center and its neighbors as visited ## immensa riduzione dei costi computazionali
                        for i in range(-neighbors, neighbors+1):  # -1, 0, 1 if n = 1
                            for j in range(-neighbors, neighbors+1):  # -1, 0, 1 if n = 1
                                if 0 <= y + i < height and 0 <= x + j < width:
                                    visited[y + i, x + j] = True

                    pbar.update(1)
    return masked_images


# load an example image file (or replace with any numpy image array)
img_path = r"C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\data\datasets\adele_test_set\extracted_images\ANGRY\image_3_ANGRY_3a30de5b8a5254fa675cc0192f56d06a.png"
img = Image.open(img_path).convert("RGB")
img_array = np.array(img)

perturbed = create_black_square_images(img_array, [35, 27, 19])

# prepare images: put original first, then all perturbed
all_images = [img_array.astype(np.uint8)] + [p.astype(np.uint8) for p in perturbed]

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