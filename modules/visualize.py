import matplotlib.pyplot as plt
import os
import time

from modules.config import SAVED_IMAGES_PATH
from modules.misc import get_timestamp

def plot_image(image, title="No Title", save_instead_of_plot=False):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    if save_instead_of_plot:
        save_path = os.path.join(SAVED_IMAGES_PATH, f"{get_timestamp()}_{title}.png")
        plt.savefig(save_path)
    else:
        plt.show()