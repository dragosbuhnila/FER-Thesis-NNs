import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import argparse

from modules.config import EMOTIONS

# Define the path to the script
create_images_script_path = os.path.join(os.path.dirname(__file__), 'occlude_dataset_offline.py')
create_h5_from_images_script_path = os.path.join(os.path.dirname(__file__), 'occluded_dataset_offline_h5.py')
count_files_in_folder_script_path = os.path.join(os.path.dirname(__file__), 'tools', 'count_files_in_folder.py')
read_h5_folder_script_path = os.path.join(os.path.dirname(__file__), 'tools', 'reading_files', 'read_h5_folder.py')

# --batch_size 16 --small_subset --show_images -d
parser = argparse.ArgumentParser(description="Run occlude_dataset_offline.py for all emotions except neutral.")
parser.add_argument("--create_images_script",               action="store_true", help="Path to the create images script")
parser.add_argument("--create_h5_script",                   action="store_true", help="Path to the create HDF5 script")
parser.add_argument("--check_images",                       action="store_true", help="Check the created images")
parser.add_argument("--check_h5",                           action="store_true", help="Check the created HDF5")
parser.add_argument("--batch_size-images",          type=int, default=16,        help="Batch size for processing")
parser.add_argument("--batch_size-h5",              type=int, default=32768,     help="Batch size for HDF5 creation")
parser.add_argument("--small_subset",                       action="store_true", help="Use a small subset of the data")
parser.add_argument("--show_images",                        action="store_true", help="Show images during processing")
parser.add_argument("--dont_parallelize_loading_landmarks", action="store_true", help="Do not parallelize loading landmarks")
parser.add_argument("-d", "--debug",                        action="store_true", help="Enable debug mode")

args = parser.parse_args()

if (not args.create_images_script) and (not args.create_h5_script) and (not args.check_images) and (not args.check_h5):
    print("No action specified. Use some of --create_images_script, --create_h5_script, --check_images, --check_h5")
    sys.exit(1)


# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/occlude_dataset_offline_run.py" --create_h5_script --create_images_script --check_images --check_h5
if __name__ == "__main__":
    if args.create_images_script:
        # Iterate over all emotions except "neutral"
        for emotion in EMOTIONS:
            if emotion.lower() == "neutral":
                continue

            # Run for both positive and negative occlusions
            for pos_or_neg in ["positive", "negative"]:
                # Build the command
                command = [
                    "python", create_images_script_path,
                    "--mismatch", emotion.lower(),
                    "--pos_or_neg", pos_or_neg,
                    "--batch_size", str(args.batch_size_images),
                ]

                if args.small_subset:
                    command.append("--small_subset")
                if args.show_images:
                    command.append("--show_images")
                if args.debug:
                    command.append("-d")
                if args.dont_parallelize_loading_landmarks:
                    command.append("--dont_parallelize_loading_landmarks")

                # Print the command for debugging
                print(f"Running: {' '.join(command)}")

                # Run the command
                subprocess.run(command, check=True)

    if args.create_h5_script:
        # After processing all emotions and occlusion types, create the HDF5 dataset
        h5_command = [
            "python", create_h5_from_images_script_path,
            "--batch_size", str(args.batch_size_h5)
        ]


        print(f"Running: {' '.join(h5_command)}")
        subprocess.run(h5_command, check=True)

    # Finally, count files in the occluded dataset folders
    print("Counting files in occluded dataset folders:")
    subprocess.run(["python", count_files_in_folder_script_path], check=True)
    print("Verifying HDF5 contents:")
    subprocess.run(["python", read_h5_folder_script_path], check=True)