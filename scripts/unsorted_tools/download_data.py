import gdown
import zipfile
import os
import shutil

STEPS = {
    "DOWNLOAD": True,
    "EXTRACT": True,
    "MOVE": True,
}

DOWNLOADS_DIR = os.path.join(".", "downloads")

URL = "https://drive.google.com/uc?id=1jTN-Q23V_QHGkeYtnlYQbYTAjdZuagDL&export=download"

if __name__ == "__main__":
    output = os.path.join(DOWNLOADS_DIR, "data.zip")
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)

    if STEPS["DOWNLOAD"]:
        gdown.download(URL, output, quiet=False)

    # Extract the ZIP file if it exists
    if os.path.exists(output):
        if STEPS["EXTRACT"]:
            print(f">> Extracting {output}...")
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall("data")
            print("<< Extraction complete.")
        if STEPS["MOVE"]:
            print(">> Moving up for cleanliness...")
            extracted_folder = os.path.join("data", "data")
            temp_destination = "data_temp"
            desired_destination = "data"
            shutil.move(extracted_folder, temp_destination)
            # rename temp_destination to desired_destination
            if os.path.exists(desired_destination):
                shutil.rmtree(desired_destination)
            shutil.move(temp_destination, desired_destination)
            print("<< Move complete.")
    else:
        print("Download failed or file not found.")