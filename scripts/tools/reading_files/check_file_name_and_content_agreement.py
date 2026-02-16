import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import re

from modules.config import OUT_ERR_DIR



def verify_filename(log_file, filename):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    # Extract the "Now running" line and the final accuracy
    now_running_line = next(line for line in lines if line.startswith("Now running"))
    final_accuracy_line = next(line for line in reversed(lines) if "categorical_accuracy" in line)
    
    # Extract settings from "Now running" line
    model_name = re.search(r"--model_name (\w+)", now_running_line).group(1)
    occlusion_ratio = re.search(r"--gen_train_occlusion_ratio ([\d.]+)", now_running_line)
    occlusion_ratio = occlusion_ratio.group(1) if occlusion_ratio else "0.5"
    unfreeze = re.search(r"--unfreeze (\d+)", now_running_line).group(1)
    long_epochs = "--long_epochs" in now_running_line
    
    # Extract final accuracy
    final_accuracy = re.search(r"categorical_accuracy: ([\d.]+)", final_accuracy_line).group(1)
    final_accuracy = f"{float(final_accuracy):.4f}".replace(".", "")
    
    # Parse filename
    filename_parts = filename.split("_")
    file_model_name = filename_parts[3]
    file_occ = filename_parts[4][3:]
    file_unf = filename_parts[5][3:]
    file_longep = "longep" in filename
    file_acc = filename_parts[-1].split(".")[0][3:]
    
    # Verify each component
    assert model_name == file_model_name, f"Model name mismatch: {model_name} != {file_model_name}"
    assert occlusion_ratio == f"0.{file_occ}", f"Occlusion ratio mismatch: {occlusion_ratio} != 0.{file_occ}"
    assert unfreeze == file_unf, f"Unfreeze mismatch: {unfreeze} != {file_unf}"
    assert long_epochs == file_longep, f"Long epochs mismatch: {long_epochs} != {file_longep}"
    assert final_accuracy == file_acc, f"Accuracy mismatch: {final_accuracy} != {file_acc}"
    
    print("Filename matches the log file contents.")



JOB_NAME = "dubh_do_training_"

filenames = [f for f in os.listdir(OUT_ERR_DIR) if JOB_NAME in f]

for filename in filenames:
    verify_filename(os.path.join(OUT_ERR_DIR, filename), filename)