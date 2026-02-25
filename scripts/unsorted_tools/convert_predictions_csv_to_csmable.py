import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Add project root to sys.path

import pandas as pd

from modules.config import OCCFT_MODELS_RESULTS_PATHS, EMOTIONS

FOLDER_PATHS = list(OCCFT_MODELS_RESULTS_PATHS.values())

if __name__ == "__main__":

    for folder_path in FOLDER_PATHS:
        input_csv_path = None
        for file_name in os.listdir(folder_path):
            if file_name.endswith("probs_ytrue_ypred.csv"):
                input_csv_path = os.path.join(folder_path, file_name)
                output_csv_path = os.path.join(folder_path, f"predictions_csmable.csv")
                output_indices_csv_path = os.path.join(folder_path, f"predictions_csmable_indices.csv")

        if input_csv_path is None:
            print(f"Skipping folder {folder_path} as it does not contain a 'probs_ytrue_ypred.csv' file.")
            continue
        else:
            print(f"Processing folder: {folder_path}")
        
        df = pd.read_csv(input_csv_path)

        # Create the required columns
        df['Image'] = df.index.map(lambda x: f"image_{x}")  # Generate image names as "image_0", "image_1", etc.
        df['True_Class'] = df['True_Label']  # Map True_Label to True_Class
        df['Predicted_Class'] = df['Predicted_Label']  # Map Predicted_Label to Predicted_Class

        # Select only the required columns
        results_df = df[['Image', 'True_Class', 'Predicted_Class']]

        # Save the results to the output CSV
        results_df.to_csv(output_csv_path, index=False)

        # Convert the True_Class and Predicted_Class columns to indices
        emotion_to_index = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        results_df['True_Class_Index'] = results_df['True_Class'].map(emotion_to_index)
        results_df['Predicted_Class_Index'] = results_df['Predicted_Class'].map(emotion_to_index)

        # Save the indices to a separate CSV file
        results_df[['Image', 'True_Class_Index', 'Predicted_Class_Index']].to_csv(output_indices_csv_path, index=False)

        print(f"Converted CSV saved to: {output_csv_path}")
