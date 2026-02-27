import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import numpy as np

from modules.misc import print_npy



parser = argparse.ArgumentParser(description="Print the contents of a .npy file to a text file.")
ACTION_OPTIONS = ["print", "size"]
parser.add_argument("--action", type=str, choices=ACTION_OPTIONS, default="size", help="Action to perform: 'print' to print contents, 'size' to print shape of the array.")
args = parser.parse_args()





if __name__ == "__main__":

    npy_file_paths = [
        "C:\\Users\\Dragos\\Roba\\Lectures\\YM2.2\\Thesis\\e Models\\results_light\\random_things\\ANGRY_Disgust.npy",
    ]

    output_file_paths = [
        f"C:\\Users\\Dragos\\Roba\\Lectures\\YM2.2\\Thesis\\e Models\\results_light\\random_things\\ANGRY_Disgust_{args.action}.npy",
    ]

    if args.action == "size":
        for npy_file_path in npy_file_paths:
            array = np.load(npy_file_path)
            print(f"Shape of {npy_file_path}: {array.shape}")
    elif args.action == "print":
        for npy_file_path, output_file_path in zip(npy_file_paths, output_file_paths):
            print_npy(npy_file_path, output_file_path)
            print(f"Printed {npy_file_path} to {output_file_path}")
    else:
        print(f"Unknown action: {args.action}. Please choose from {ACTION_OPTIONS}.")