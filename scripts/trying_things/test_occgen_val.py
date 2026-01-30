# Don't run this script directly. Provide arguments!
# & "C:/Users/Dragos/.conda/envs/fer-thesis/python.exe" "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/test_occgen_val.py" -o 0.5  -f lines -l true -r false -m true -a 0.2



import os; import sys
from tqdm import tqdm; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import argparse
import pstats

from modules.mask import MASKING_FUNCTIONS



# Argument parser setup
parser = argparse.ArgumentParser(description="Test data generator with occlusion layer.")
parser.add_argument("-o", "--occlusion_probability", type=float, required=True, help="Probability of occlusion (float).")
parser.add_argument("-f", "--masking_function", type=str, required=True, choices=MASKING_FUNCTIONS.keys(), help=f"Masking function to use ({', '.join(MASKING_FUNCTIONS.keys())}).")
parser.add_argument("-l", "--label_smoothing",  type=str, required=True, default=False, help="Whether to use label smoothing in training (boolean).")
parser.add_argument("-r", "--redirect_output",  type=str, required=True, default=False, help="Whether to redirect output to a log file (boolean).")
parser.add_argument("-m", "--mismatch",         type=str, required=True, default=False, help="Whether to use mismatched occlusions (boolean).")
parser.add_argument("-a", "--matching_amount",  type=float, required=False, help="Amount of matching for occlusions (float). Exaple: 0.2 is 20%, i.e. out of 50 images 10 will be matching, the rest will be of some mismatch type (every 4)")

args = parser.parse_args()

args.label_smoothing =  True if str(args.label_smoothing).lower()   in ['true', '1', 'yes'] else False
args.redirect_output =  True if str(args.redirect_output).lower()   in ['true', '1', 'yes'] else False
args.mismatch =         True if str(args.mismatch).lower()          in ['true', '1', 'yes'] else False
if args.mismatch == True:
    if args.matching_amount is None:
        raise ValueError("If mismatch is True, matching_amount must be provided as a float.")
    if not (0.0 <= args.matching_amount <= 1.0):
        raise ValueError("matching_amount must be between 0.0 and 1.0.")
elif args.mismatch == False:
    if args.matching_amount is not None:
        print("Warning: matching_amount is provided but mismatch is False. Ignoring matching_amount.")

from modules.config import ADELE_TEST_SET_H5_PATH, BOSPHORUS_TEST_HQ_H5_PATH, CONSOLE_OUTPUTS_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, GLOBALS
from modules.data__load import load_data_generators, load_valid_generator
from modules.visualize import plot_image



# ============================= MACROS ============================

# __________________-DATASETS-_________________
#        1161 x 1161               128 x 128                   128 x 128
# BOSPHORUS_TEST_HQ_H5_PATH, ADELE_TEST_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = ADELE_TEST_SET_H5_PATH  
TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH

USE_PROFILER = False
STOP_AFTER_IMAGES = 0

# __________________-REDIRECTION-_________________
REDIRECT_OUTPUT = args.redirect_output
LOG_FILE = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__test_occgens_all__console_output.txt")

print("SETTINGS selected:")
print(f"  TEST_SET_PATH = {TEST_SET_PATH}")
print(f"  TRAINVAL_SET_PATH = {TRAINVAL_SET_PATH}")
print(f"  USE_PROFILER = {USE_PROFILER}")
print(f"  STOP_AFTER_IMAGES = {STOP_AFTER_IMAGES}")
print(f"  LOG_FILE = {LOG_FILE}")
print()
print(f"  OCCLUSION_PROBABILITY = {args.occlusion_probability}")
print(f"  MASKING_FUNCTION = {args.masking_function}")
print(f"  LABEL_SMOOTHING = {args.label_smoothing}")
print(f"  REDIRECT_OUTPUT = {REDIRECT_OUTPUT}")
print(f"  MISMATCH = {args.mismatch}")
print()

# ========================== END OF MACROS ========================



if USE_PROFILER:
    import cProfile

if REDIRECT_OUTPUT:
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = open(LOG_FILE, "w")
    sys.stderr = sys.stdout



if __name__ == "__main__":
    # 0) Access arguments
    occlusion_probability = args.occlusion_probability
    masking_function = args.masking_function
    use_label_smoothing = args.label_smoothing

    # 1) First time run with occlusions      
    val_generator = load_valid_generator(
        train_path=TRAINVAL_SET_PATH,
        occlusion_probability=occlusion_probability,
        masking_function_name=masking_function,
        mismatch=args.mismatch,
        matching_amount=args.matching_amount,
    )

    start_time = time.time() 
    if USE_PROFILER:
        profiler = cProfile.Profile()
        profiler.enable()

    print(f"Testing generator: val_generator")
    try:
        first_batch = True
        for batch in tqdm(val_generator, desc=f"Going through data generator val_generator"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                X_batch, y_batch = batch[0], batch[1]
            else:
                raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
            if first_batch:
                print(f"First: Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
                first_batch = False

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
            # break  # Just test one batch for timing purposes  
        print(f"Last: Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
    except Exception as e:
        print(f"An error occurred while processing generator val_generator, printing info so far:")
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
    if len(GLOBALS["UNLANDMARKABLE_IMAGES_LIST"]) > 0:
        print("The following images encountered an issue with the detection of landmarks, or of the face:")
        for hash in GLOBALS["UNLANDMARKABLE_IMAGES_LIST"]:
            print(f" - {hash}")
        print()

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

