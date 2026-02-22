import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", '..')))
import time
import argparse
import cProfile
import pstats

from modules.mask import MASKING_FUNCTIONS

# Argument parser setup
parser = argparse.ArgumentParser(description="Test data generator with occlusion layer.")
parser.add_argument("-o", "--occlusion_probability", type=float, required=True, help="Probability of occlusion (float).")
parser.add_argument("-m", "--masking_function", type=str, required=True, choices=MASKING_FUNCTIONS.keys(), help=f"Masking function to use ({', '.join(MASKING_FUNCTIONS.keys())}).")
args = parser.parse_args()

from modules.config import ADELE_TEST_SET_H5_PATH, BOSPHORUS_TEST_HQ_H5_PATH, CONSOLE_OUTPUTS_PATH, GLOBALS
from modules.data import load_test_generator



# ============================= MACROS ============================

#        1161 x 1161               128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH 
DATASET = ADELE_TEST_SET_H5_PATH     

USE_PROFILER = False
REDIRECT_OUTPUT = True
LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"test_occgen_test__{time.strftime('%Y%m%d-%H%M%S')}__console_output.txt")

# ========================== END OF MACROS ========================



if USE_PROFILER:
    import cProfile

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout



# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/test_occgen_test.py" -o 1.0 -m lines
if __name__ == "__main__":
    # 0) Access arguments
    occlusion_probability = args.occlusion_probability
    masking_function = MASKING_FUNCTIONS[args.masking_function]
    print(f"Testing data generator with occlusion_probability={occlusion_probability}, masking_function={args.masking_function}")

    # 1) First time run with occlusions
    start_time = time.time() 
    if USE_PROFILER:
        profiler = cProfile.Profile()
        profiler.enable()
    test_generator = load_test_generator(DATASET, occlusion_probability=occlusion_probability, masking_function=masking_function)

    for batch in test_generator:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X_batch, y_batch, batch_paths = batch[0], batch[1], batch[2]
        else:
            raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
        
        print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}, Batch paths: {batch_paths if batch_paths[0] is not None else f'[None*{len(batch_paths)}]'}")

        for image in X_batch:
            pass
            # print(f"Image shape: {image.shape}")
            # plot_image(image)
    
    if USE_PROFILER:
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(20)
        stats.print_stats()
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for generator with occlusion: {elapsed_time:.6f} seconds")
    print(f"Loaded a total of {GLOBALS['TOTAL_IMAGES_LANDMARKS_LOADED']} images landmarks, saved {GLOBALS['TOTAL_IMAGES_LANDMARKS_SAVED']} images landmarks.")

    # # 2) Second time run without occlusions
    # start_time = time.time()
    # test_generator = load_data_generator(DATASET, 'test', 0.0)
    
    # for batch in test_generator:
    #     if isinstance(batch, (list, tuple)) and len(batch) >= 2:
    #         X_batch, y_batch, batch_paths, batch_x_hashes = batch[0], batch[1], batch[2], batch[3]
    #     else:
    #         raise ValueError("test_generator must yield (X_batch, y_batch) tuples")

    #     print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}, Batch paths: {batch_paths.shape if batch_paths is not None else 'None'}, Batch hashes: {batch_x_hashes.shape}")

    #     for image in X_batch:
    #         pass
    #         # print(f"Image shape: {image.shape}")
    #         # plot_image(image)

    # end_time = time.time()  # Record the end time
    # elapsed_time = end_time - start_time  # Calculate elapsed time
    # print(f"Elapsed time for generator without occlusion: {elapsed_time:.6f} seconds")

