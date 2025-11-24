import os; import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import argparse
import pstats

from modules.mask import MASKING_FUNCTIONS

# Argument parser setup
parser = argparse.ArgumentParser(description="Test data generator with occlusion layer.")
parser.add_argument("-o", "--occlusion_probability", type=float, required=True, help="Probability of occlusion (float).")
parser.add_argument("-m", "--masking_function", type=str, required=True, choices=MASKING_FUNCTIONS.keys(), help=f"Masking function to use ({', '.join(MASKING_FUNCTIONS.keys())}).")
parser.add_argument("-l", "--label_smoothing", type=bool, default=False, help="Whether to use label smoothing in training (boolean).")
parser.add_argument("-r", "--redirect_output", type=bool, default=False, help="Whether to redirect output to a log file (boolean).")
args = parser.parse_args()

from modules.config import ADELE_TEST_SET_H5_PATH, BOSPHORUS_TEST_HQ_H5_PATH, CONSOLE_OUTPUTS_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, GLOBALS
from modules.data import load_data_generators
from modules.visualize import plot_image



# ============================= MACROS ============================

#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = ADELE_TEST_SET_H5_PATH  
TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH

USE_PROFILER = False
STOP_AFTER_IMAGES = 0
REDIRECT_OUTPUT = args.redirect_output

LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__test_occgens_all__console_output.txt")

print("SETTINGS selected:")
print(f"  TEST_SET_PATH = {TEST_SET_PATH}")
print(f"  TRAINVAL_SET_PATH = {TRAINVAL_SET_PATH}")
print(f"  USE_PROFILER = {USE_PROFILER}")
print(f"  STOP_AFTER_IMAGES = {STOP_AFTER_IMAGES}")
print(f"  REDIRECT_OUTPUT = {REDIRECT_OUTPUT}")
print(f"  LOG_FILE = {LOG_FILE}")

# ========================== END OF MACROS ========================



if USE_PROFILER:
    import cProfile

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout



# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/test_occgens_all.py" -o 1.0  -m lines -l true -r true
if __name__ == "__main__":
    # 0) Access arguments
    occlusion_probability = args.occlusion_probability
    masking_function = MASKING_FUNCTIONS[args.masking_function]
    use_label_smoothing = args.label_smoothing
    print(f"Testing data generators with occlusion_probability={occlusion_probability}, masking_function={args.masking_function}, use_label_smoothing={use_label_smoothing}")

    # 1) First time run with occlusions      
    train_generator, val_generator, test_generator, initial_bias = load_data_generators(TRAINVAL_SET_PATH, TEST_SET_PATH, occlusion_probability, masking_function, use_label_smoothing)

    for generator, name in [(train_generator, "train"), (val_generator, "validation"), (test_generator, "test")]:
        start_time = time.time() 
        if USE_PROFILER:
            profiler = cProfile.Profile()
            profiler.enable()

        print(f"Testing generator: {name}")
        try:
            for batch in generator:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X_batch, y_batch, batch_paths = batch[0], batch[1], batch[2]
                else:
                    raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
                print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}, Batch paths: {batch_paths if batch_paths[0] is not None else f'[None*{len(batch_paths)}]'}")

                i = 0
                for image in X_batch:
                    if STOP_AFTER_IMAGES == 0:
                        continue
                    else:
                        print(f"Image shape: {image.shape}")
                        plot_image(image)
                        i += 1
                        if i >= STOP_AFTER_IMAGES:
                            break
                break  # Just test one batch for timing purposes  
        except Exception as e:
            print(f"An error occurred while processing generator {name}, printing info so far:")
            print(f"Loaded a total of {GLOBALS['TOTAL_IMAGES_LANDMARKS_LOADED']} images landmarks, saved {GLOBALS['TOTAL_IMAGES_LANDMARKS_SAVED']} images landmarks.")
            print("The following images encountered an issue with the detection of landmarks, or of the face:")
            for hash in GLOBALS["UNLANDMARKABLE_IMAGES_LIST"]:
                print(f" - {hash}")
            raise e

        if USE_PROFILER:
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(20)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Elapsed time for generator with occlusion: {elapsed_time:.6f} seconds")
        print(f"Loaded a total of {GLOBALS['TOTAL_IMAGES_LANDMARKS_LOADED']} images landmarks, saved {GLOBALS['TOTAL_IMAGES_LANDMARKS_SAVED']} images landmarks.")
        if len(GLOBALS["UNLANDMARKABLE_IMAGES_LIST"] > 0):
            print("The following images encountered an issue with the detection of landmarks, or of the face:")
            for hash in GLOBALS["UNLANDMARKABLE_IMAGES_LIST"]:
                print(f" - {hash}")
        

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

