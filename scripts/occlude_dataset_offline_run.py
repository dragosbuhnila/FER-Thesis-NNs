import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess

from modules.config import EMOTIONS

# Define the path to the script
script_path = os.path.join(os.path.dirname(__file__), 'occlude_dataset_offline.py')

# Define the batch size
batch_size = 16

# Iterate over all emotions except "neutral"
for emotion in EMOTIONS:
    if emotion.lower() == "neutral":
        continue

    # Run for both positive and negative occlusions
    for pos_or_neg in ["positive", "negative"]:
        # Build the command
        command = [
            "python", script_path,
            "--mismatch", emotion.lower(),
            "--pos_or_neg", pos_or_neg,
            "--batch_size", str(batch_size)
        ]

        # Print the command for debugging
        print(f"Running: {' '.join(command)}")

        # Run the command
        subprocess.run(command, check=True)