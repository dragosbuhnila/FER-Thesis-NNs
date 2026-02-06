# Don't run this script directly. Provide arguments!

import os; import sys
from tqdm import tqdm; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import argparse
import pstats

from modules.config import CONSOLE_OUTPUTS_PATH, EMOTIONS, OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH, ORIGINAL_TRAIN_VAL_SET_H5_PATH, GLOBALS
from modules.data__load import load_offline_data_generators
from modules.data import refresh_show_flags
from modules.visualize import plot_image
from modules.misc import Tee



# ============================ ARGUMENTS ============================


LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{time.strftime('%Y%m%d-%H%M%S')}__test_occgens_all__console_output.txt")

ORIGINAL_TRAINVAL_SET_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
OCCLUDED_TRAINVAL_SET_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH
TEST_SET_PATH = OCCLUDED_TEST_SET_H5_PATH

HOW_MANY_BATCHES_TO_TEST = 1  # Set to None to test all batches


parser = argparse.ArgumentParser(description='Generate occluded dataset offline and save to HDF5')
parser.add_argument("--batch_size",                type=int, default=32, help="Batch size for data generators")
parser.add_argument("--small_subset",               action='store_true', help="Use a small subset of the data for quick testing")
parser.add_argument("--dont_augment",               action='store_true', help="Do not apply augmentations to the data")
parser.add_argument("--show_loader_images_b4aug",   action='store_true', help="Show images from the data loader before augmentations")
parser.add_argument("--show_loader_images_final",   action='store_true', help="Show images from the data loader after all augmentations")
parser.add_argument("--use_profiler",               action='store_true', help="Use cProfile to profile the data loading process")
parser.add_argument("--redirect_output",            action='store_true', help="Redirect console output to a log file. This does not mean terminal won't show stderr/out")
parser.add_argument("-s", "--save_loader_images_instead_of_plot",         action='store_true', help="Save images from the data loader instead of plotting them (useful for debugging in non-interactive environments and for keeping a record of the images that were shown during debugging)")
args = parser.parse_args()


GLOBALS["DATALOADER_SHOW_IMAGES_B4AUG"] = args.show_loader_images_b4aug
GLOBALS["DATALOADER_SHOW_IMAGES_FINAL"] = args.show_loader_images_final
GLOBALS["DATALOADER_SHOW_IMAGES_SAVE_INSTEAD_OF_PLOT"] = args.save_loader_images_instead_of_plot
refresh_show_flags()


print("================================== SETTINGS ==================================", flush=True)
print(f"CONSTANTS: ")
print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"\tORIGINAL_TRAINVAL_SET_PATH: {ORIGINAL_TRAINVAL_SET_PATH}")
print(f"\tOCCLUDED_TRAINVAL_SET_PATH: {OCCLUDED_TRAINVAL_SET_PATH}")
print(f"\tTEST_SET_PATH: {TEST_SET_PATH}")
print("ARGS: ")
print(f"\tbatch_size: {args.batch_size}")
print(f"\tsmall_subset: {args.small_subset}")
print(f"\tdont_augment: {args.dont_augment}")
print(f"\tshow_loader_images_b4aug: {args.show_loader_images_b4aug}")
print(f"\tshow_loader_images_final: {args.show_loader_images_final}")
print(f"\tuse_profiler: {args.use_profiler}")
print(f"\tredirect_output: {args.redirect_output}")
print(f"\tsave_loader_images_instead_of_plot: {args.save_loader_images_instead_of_plot}")
print("GLOBALS: ")
for key, value in GLOBALS.items():
    print(f"\t{key}: {value}")
print("===============================================================================", flush=True)


if args.use_profiler:
    import cProfile

if args.redirect_output:
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH) 



# ========================== END OF MACROS ========================


# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/trying_things/test_offline_occgens_all.py" --batch_size 32 --small_subset --show_loader_images_b4aug --show_loader_images_final
if __name__ == "__main__":
    # 1) First time run with occlusions      
    train_generator, val_generator, test_generator, initial_bias = load_offline_data_generators(
                                                            # Paths ---------------------------------------------------
                                                            original_trainval_path=ORIGINAL_TRAINVAL_SET_PATH,
                                                            occluded_trainval_path=OCCLUDED_TRAINVAL_SET_PATH,
                                                            occluded_test_path=TEST_SET_PATH,
                                                            # Occlusion parameters ------------------------------------

                                                            # Command line args for working ---------------------------
                                                            small_subset=args.small_subset,
                                                            batch_size=args.batch_size,
                                                            dont_augment=args.dont_augment,
                                                        )

    for generator, name in [(train_generator, "train"), (val_generator, "validation"), (test_generator, "test")]:
        start_time = time.time() 
        if args.use_profiler:
            profiler = cProfile.Profile()
            profiler.enable()

        print(f"Testing generator: {name}")
        try:
            first_batch = True
            batches_tested = 0
            for batch in tqdm(generator, desc=f"Going through data generator {name}"):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X_batch, y_batch = batch[0], batch[1]
                else:
                    raise ValueError("test_generator must yield (X_batch, y_batch) tuples")
                if first_batch:
                    print(f"First: Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
                    first_batch = False

                i = 0
                for image in X_batch:
                    nop_thing = 3

                batches_tested += 1
                if HOW_MANY_BATCHES_TO_TEST is not None and batches_tested >= HOW_MANY_BATCHES_TO_TEST:
                    print(f"Reached the limit of {HOW_MANY_BATCHES_TO_TEST} batches to test, stopping.")
                    break
            print(f"Last: Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
        except Exception as e:
            print(f"An error occurred while processing generator {name}, printing info so far:")
            print(f"Loaded a total of {GLOBALS['TOTAL_IMAGES_LANDMARKS_LOADED']} images landmarks, saved {GLOBALS['TOTAL_IMAGES_LANDMARKS_SAVED']} images landmarks.")
            print("The following images encountered an issue with the detection of landmarks, or of the face:")
            for hash in GLOBALS["UNLANDMARKABLE_IMAGES_LIST"]:
                print(f" - {hash}")
            raise e

        if args.use_profiler:
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

    print(f"Finished testing data generators with occlusions. Reprinting GLOBALS:")
    print(GLOBALS)
        

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

