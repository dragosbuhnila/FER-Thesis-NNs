import os
import sys
from rich.console import Console
from rich.table import Table

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import h5py

from modules.config import OCCLUDED_TEST_SET_H5_PATH, EMOTIONS

if __name__ == '__main__':
    console = Console()

    with h5py.File(OCCLUDED_TEST_SET_H5_PATH, 'r') as f:
        paths = f['paths'][:]

        occluding_emotions_counts = {emotion_name.upper(): 0 for emotion_name in EMOTIONS}
        gt_emotions_counts = {emotion_name.upper(): 0 for emotion_name in EMOTIONS}
        positive_negative_counts = {'POSITIVE': 0, 'NEGATIVE': 0}
        match_mismatch_counts = {'MATCH': 0, 'MISMATCH': 0}

        # Extract the gt emotion, occlusion emotion, positive/negative occlusion, and match/mismatch info from the paths
        for path in paths:
            path_str = path.decode('utf-8')         # Decode bytes to string
            path_str = path_str.split('\\')[-1]     # Get the last part of the path (the filename)
            filename = path_str.strip(".png")       # Remove the file extension 
            filename = filename.replace("__", "_")  # Replace double underscores with single underscores for easier splitting
            filename = filename.replace("-", "_")   # Replace dashes with underscores for easier splitting

            parts = filename.split('_')  # Split by '__'
            gt_emotion = parts[2].upper()
            occlusion_emotion = parts[6].upper()
            positive_negative = parts[5].upper()
            match_mismatch = parts[7].upper()

            if gt_emotion not in EMOTIONS:
                raise ValueError(f"Unexpected GT emotion: {gt_emotion}")
            if occlusion_emotion not in EMOTIONS:
                raise ValueError(f"Unexpected occlusion emotion: {occlusion_emotion}")
            if positive_negative not in ['POSITIVE', 'NEGATIVE']:
                raise ValueError(f"Unexpected positive/negative value: {positive_negative}")
            if match_mismatch not in ['MATCH', 'MISMATCH']:
                raise ValueError(f"Unexpected match/mismatch value: {match_mismatch}")
            if gt_emotion == occlusion_emotion and match_mismatch != 'MATCH':
                raise ValueError(f"GT emotion and occlusion emotion are the same but match/mismatch is not 'MATCH': {filename}")

            gt_emotions_counts[gt_emotion] += 1
            occluding_emotions_counts[occlusion_emotion] += 1
            positive_negative_counts[positive_negative] += 1
            match_mismatch_counts[match_mismatch] += 1

        # Display results using rich tables
        def display_table(title, data):
            table = Table(title=title)
            table.add_column("Category", justify="left", style="cyan", no_wrap=True)
            table.add_column("Count", justify="right", style="magenta")
            for key, value in data.items():
                table.add_row(key, str(value))
            console.print(table)

        display_table("GT Emotion Counts", gt_emotions_counts)
        display_table("Occluding Emotion Counts", occluding_emotions_counts)
        display_table("Positive/Negative Counts", positive_negative_counts)
        display_table("Match/Mismatch Counts", match_mismatch_counts)