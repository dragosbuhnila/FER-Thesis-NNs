import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


from modules.misc import print_npy



if __name__ == "__main__":

    npy_file_paths = [
        "C:\\Users\\Dragos\\Roba\\Lectures\\YM2.2\\Thesis\\e Models\\results_light\\debugging_hashes\\20251127-145756__indices_to_hashes.npy",
    ]

    output_file_paths = [
        "C:\\Users\\Dragos\\Roba\\Lectures\\YM2.2\\Thesis\\e Models\\results_light\\debugging_hashes\\20251127-145756__indices_to_hashes_readable.txt",
    ]

    for npy_file_path, output_file_path in zip(npy_file_paths, output_file_paths):
        print_npy(npy_file_path, output_file_path)
        print(f"Printed {npy_file_path} to {output_file_path}")