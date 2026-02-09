import os
import matplotlib.pyplot as plt
import time

from modules.config import SAVED_IMAGES_PATH, GLOBALS
from modules.misc import get_timestamp

if GLOBALS["START_TIME"] is None:
    GLOBALS["START_TIME"] = get_timestamp()
    START_TIME = time.time()

def plot_image(image, title="No Title", save_instead_of_plot=False):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    if save_instead_of_plot:
        # Replace spaces and slashes to avoid issues in filenames, then remove special characters
        save_safe_title = title.replace(" ", "_").replace("/", "_")
        save_safe_title = ''.join(c for c in save_safe_title if c.isalnum() or c in ['_', '-'])
        save_path = os.path.join(SAVED_IMAGES_PATH, str(START_TIME), f"{get_timestamp()}_{save_safe_title}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()