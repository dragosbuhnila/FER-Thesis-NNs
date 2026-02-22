import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import re

from modules.config import OUT_ERR_DIR



def verify_filename(log_file, filename):
    print(f"Verifying {filename}...")
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    # Extract the "Now running" line and the final accuracy
    now_running_line = next(line for line in lines if line.startswith("Now running"))
    final_accuracy_line = next(line for line in reversed(lines) if "categorical_accuracy" in line)
    
    # Extract settings from "Now running" line
    model_name = re.search(r"--model_name (\w+)", now_running_line).group(1)
    occlusion_ratio = re.search(r"--gen_train_occlusion_ratio ([\d.]+)", now_running_line)
    occlusion_ratio = occlusion_ratio.group(1) if occlusion_ratio else "0.8"
    unfreeze = re.search(r"--unfreeze (\d+)", now_running_line).group(1)
    long_epochs = "--long_epochs" in now_running_line
    
    # Extract final accuracy
    final_accuracy = re.search(r"categorical_accuracy: ([\d.]+)", final_accuracy_line).group(1)
    final_accuracy = f"{float(final_accuracy):.4f}".replace("0.", "")
    
    # Parse filename
    filename = filename.strip(".log")  # Remove the .log extension
    filename = filename.replace(JOB_NAME, "")  # Remove the job name prefix

    parts = filename.split("_")
    file_id = int(parts[0])  # The ID is the first part of the filename
    file_model_name = parts[1]  # The model name comes after "do_training_"

    file_longep = "longep" in parts  # True if "longep" is in the parts
    file_occ = next((p for p in parts if p.startswith("occ")), None)
    file_occ = file_occ.split("occ")[1] if file_occ else None  # Extract the value after "occ"
    file_unf = next((p for p in parts if p.startswith("unf")), None)
    file_unf = file_unf.split("unf")[1] if file_unf else None  # Extract the value after "unf"
    file_acc = next((p for p in parts if p.startswith("acc")), None)
    file_acc = file_acc.split("acc")[1] if file_acc else None  # Extract the value after "acc"
    
    # Verify each component
    assert model_name == file_model_name, f"Model name mismatch: {model_name} != {file_model_name}"
    assert occlusion_ratio == f"0.{file_occ}", f"Occlusion ratio mismatch: {occlusion_ratio} != 0.{file_occ}"
    assert unfreeze == file_unf, f"Unfreeze mismatch: {unfreeze} != {file_unf}"
    if file_id < 1163444 and file_id > 1163451:
        assert long_epochs == file_longep, f"Long epochs mismatch: {long_epochs} != {file_longep}"
    assert final_accuracy == file_acc, f"Accuracy mismatch: {final_accuracy} != {file_acc}"
    
    print("Filename matches the log file contents.", flush=True)



JOB_NAME = "dbuh_do_training_"


if __name__ == "__main__":
    print("Starting verification of log files...")

    filenames = [f for f in os.listdir(OUT_ERR_DIR) if JOB_NAME in f]
    filenames.sort()
    print(f"Found {len(filenames)} log files to verify.")

    for filename in filenames:
        verify_filename(os.path.join(OUT_ERR_DIR, filename), filename)