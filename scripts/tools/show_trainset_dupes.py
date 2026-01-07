import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import h5py

from modules.config import ORIGINAL_TRAIN_VAL_SET_H5_PATH



DATASET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH



CONFLICT_INDICES = {
    "4f50f6cba30511ed1a5731121979986c": ("X_train", 526 , 527),        
    "1cd575ac2bb8e82ef46ad3758d64b308": ("X_train", 5526, 5527),      
    "015958af0120539c5a911df3ad77f6f8": ("X_train", 5136, 21269),      
    "6b3e14e0646a97eb1ff84f66f4896d96": ("X_train", 5137, 21270),      
}


if __name__ == "__main__":
    # 1) open the h5 file
        with h5py.File(DATASET_PATH, 'r') as h5_file:
            images = dict()
            for key in h5_file.keys():
                if key.startswith('X_'):
                    images[key] = h5_file[key][...]

            for hash, (key, idx1, idx2) in CONFLICT_INDICES.items():
                img1 = images[key][idx1]
                img2 = images[key][idx2]
                # Display the images side by side for comparison
                import matplotlib.pyplot as plt

                plt.figure(figsize=(8, 4))
                plt.suptitle(f'Comparing images that conflict on hash {hash}')

                plt.subplot(1, 2, 1)
                plt.title(f'Image at index {key}_{idx1}')
                plt.imshow(img1)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title(f'Image at index {key}_{idx2}')
                plt.imshow(img2)
                plt.axis('off')

                plt.show()
