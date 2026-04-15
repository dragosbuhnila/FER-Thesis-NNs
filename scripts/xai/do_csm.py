import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
from typing import Tuple, Union
from scipy.interpolate import griddata
import pandas as pd

from modules.misc import Tee, extract_image_index_original_or_180rotated, get_timestamp
from modules.config import ADELE_180ROTATED_TEST_SET_IMAGES_PATH, ADELE_TEST_SET_IMAGES_PATH, CANONICAL_FACES_CUT, CONSOLE_OUTPUTS_PATH, EMOTIONS, LANDMARKER_MODEL_PATH, OCCFT_MODELS_RESULTS_PATHS, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH, SAVED_IMAGES_PATH
from modules.misc import TEST_SET_CHOICES, TEST_SET_PATHS, get_test_set_name_from_run_name
from modules.config import (
    ALL_MODELS_PATHS,
    CONSOLE_OUTPUTS_PATH,
    SAVED_IMAGES_PATH,
    EMOTIONS,
)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    
    return None, None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def generate_csm(input_saliencies_path, results_csv_path, test_set_images_path, results_path,
                 extract_image_index_function,
                 limite, minimo, ext, solo_pos, colore, al,
                 save_heatmap, save_img, visualize):

    base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # TEST_SET_IMAGES_PATH
    images_path_norm = test_set_images_path
    canonical_faces_path = CANONICAL_FACES_CUT

    lim = limite

    # Load the csv file
    df = pd.read_csv(results_csv_path)
    print(df.head())

    emotions = df['True_Class'].unique()
    print(emotions)


    # creare un subplot 7x7
    fig, axs = plt.subplots(7, 7, figsize=(20, 20))
    fig.suptitle('Canonical Model Saliency Maps', fontsize=28)
    fig.suptitle('Predicted', fontsize=28)
    fig.supylabel('True', fontsize=28)



    # nascondere gli assi
    for i in range(7):
        for j in range(7):
            axs[i, j].axis('off')

    axs[0, 0].text(-0.1, 0.5, 'Anger', va='center', ha='right', rotation=0, transform=axs[0, 0].transAxes, fontsize=20)
    axs[1, 0].text(-0.1, 0.5, 'Disgust', va='center', ha='right', rotation=0,transform=axs[1, 0].transAxes, fontsize=20)
    axs[2, 0].text(-0.1, 0.5, 'Fear', va='center', ha='right', rotation=0,transform=axs[2, 0].transAxes, fontsize=20)
    axs[3, 0].text(-0.1, 0.5, 'Happiness', va='center', ha='right', rotation=0,transform=axs[3, 0].transAxes, fontsize=20)
    axs[4, 0].text(-0.1, 0.5, 'Neutral', va='center', ha='right', rotation=0,transform=axs[4, 0].transAxes, fontsize=20)
    axs[5, 0].text(-0.1, 0.5, 'Sadness', va='center', ha='right', rotation=0,transform=axs[5, 0].transAxes, fontsize=20)
    axs[6, 0].text(-0.1, 0.5, 'Surprise', va='center', ha='right', rotation=0,transform=axs[6, 0].transAxes, fontsize=20)

    axs[0, 0].set_title('Anger', fontsize=20)
    axs[0, 1].set_title('Disgust', fontsize=20)
    axs[0, 2].set_title('Fear', fontsize=20)
    axs[0, 3].set_title('Happiness', fontsize=20)
    axs[0, 4].set_title('Neutral', fontsize=20)
    axs[0, 5].set_title('Sadness', fontsize=20)
    axs[0, 6].set_title('Surprise', fontsize=20)


    riga = 0
    colonna = 0
    
    emotions = EMOTIONS


    for emotion in emotions:
        # list of images for each emotion
        images_filenames = os.listdir(os.path.join(images_path_norm, emotion))
        print(emotion)
        
        face = mp.Image.create_from_file(os.path.join(canonical_faces_path, f"{emotion}.png"))
        Fwidth = face.width
        Fheight = face.height
        Fdetection_result = detector.detect(face)
        face_landmarks_list = Fdetection_result.face_landmarks

        canonical_heatmap = np.zeros((Fheight,Fwidth))
        canonical_heatmap_w = np.zeros((Fheight,Fwidth))
        
        tot_w = 0
        tot = 0
        tot_Anger = 0
        tot_Disgust = 0
        tot_Fear = 0
        tot_Happiness = 0
        tot_Neutral = 0
        tot_Sadness = 0
        tot_Surprise = 0
        CSM_Anger = np.zeros((Fheight,Fwidth))
        CSM_Disgust = np.zeros((Fheight,Fwidth))
        CSM_Fear = np.zeros((Fheight,Fwidth))
        CSM_Happiness = np.zeros((Fheight,Fwidth))
        CSM_Neutral = np.zeros((Fheight,Fwidth))
        CSM_Sadness = np.zeros((Fheight,Fwidth))
        CSM_Surprise = np.zeros((Fheight,Fwidth))


        for image_filename in images_filenames:
            # extract_image_index_original_or_180rotated(
            image_index = extract_image_index_function(image_filename)

            heatmap = np.load(os.path.join(input_saliencies_path, emotion, f'image_{image_index}.npy'))

            # control the true class abd the predicted class
            predicted_class = df[df['Image'] == f"image_{image_index}"]['Predicted_Class'].values[0]

            # img in stringa
            image_filename = str(image_filename)
            
            images_path = images_path_norm

            image = mp.Image.create_from_file(os.path.join(images_path, emotion, f'{image_filename}'))
            Iwidth = image.width
            Iheight = image.height

            Idetection_result = detector.detect(image)

            
            image_landmarks_list = Idetection_result.face_landmarks
            # se il volto non è stato rilevato, segnala l'errore
            if len(image_landmarks_list) == 0:
                print(f"Face not detected in {image_filename}")
                continue
            
            heatmap_values = []
            for idx in range(len(image_landmarks_list[0])):
                image_landmarks = image_landmarks_list[0][idx]
                if image_landmarks == None:
                    print(f"Landmark {idx} not detected in {image_filename}")
                    continue
                x, y = _normalized_to_pixel_coordinates(image_landmarks.x, image_landmarks.y, Iwidth, Iheight)
                if x == None:
                    z = 0
                    heatmap_values.append(z)
                    continue
                # assegnare a z il valore di heatmap[x, y]
                z = heatmap[y, x]
                heatmap_values.append(z)

            # assegna a ogni landmark del volto il valore di z
            #crea una matrice con le coordinate e i valori di heatmap
            num_landmarks = len(face_landmarks_list[0])
            coordinates_face = np.zeros((num_landmarks, 3))
            for idx in range(len(face_landmarks_list[0])):
                face_landmarks = face_landmarks_list[0][idx]
                x, y = _normalized_to_pixel_coordinates(face_landmarks.x, face_landmarks.y, Fwidth, Fheight)
                z = heatmap_values[idx]
                coordinates_face[idx,:] = np.array([x, y, z])

            # create an array with the coordinates X E Y of the face
            points = coordinates_face[:, 0:2]   
                
            grid_x, grid_y = np.mgrid[0:Fwidth:75j, 0:Fheight:85j]
            grid_z = griddata(points,  coordinates_face[:, 2], (grid_x, grid_y), method='cubic')
            # plot the scaled heatmap on face
            #face = cv2.imread(os.path.join("Canonical faces", f"{emotion}.png"))
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            grid_z[grid_z < 0] = 0
            grid_z = grid_z.T
            # se grid_z è nan visualizza l'immagine
            if np.isnan(grid_z).all():
                print(f"NaN in {image_filename}")
                continue
            

            if emotion != predicted_class:
                tot_w += 1
                canonical_heatmap_w = canonical_heatmap_w + grid_z
                if predicted_class == 'ANGRY':
                    CSM_Anger = CSM_Anger + grid_z
                    tot_Anger += 1
                elif predicted_class == 'DISGUST':
                    CSM_Disgust = CSM_Disgust + grid_z
                    tot_Disgust += 1
                elif predicted_class == 'FEAR':
                    CSM_Fear = CSM_Fear + grid_z
                    tot_Fear += 1
                elif predicted_class == 'HAPPY':
                    CSM_Happiness = CSM_Happiness + grid_z
                    tot_Happiness += 1
                elif predicted_class == 'NEUTRAL':
                    CSM_Neutral = CSM_Neutral + grid_z
                    tot_Neutral += 1
                elif predicted_class == 'SAD':
                    CSM_Sadness = CSM_Sadness + grid_z
                    tot_Sadness += 1
                elif predicted_class == 'SURPRISE':
                    CSM_Surprise = CSM_Surprise + grid_z
                    tot_Surprise += 1
            else:
                tot += 1
                canonical_heatmap = canonical_heatmap + grid_z

        face = cv2.imread(os.path.join(canonical_faces_path, f"{emotion}.png"))
        

        if tot == 0:
            axs[riga, colonna].imshow(face)
            
            axs[riga, colonna].axis('off')
        else:
            canonical_heatmap = canonical_heatmap / tot
           
            if ext == True:
                canonical_heatmap = canonical_heatmap/np.nanmax(np.abs(canonical_heatmap))
                if solo_pos:
                    canonical_heatmap[canonical_heatmap < 0] = 0
            else: 
                canonical_heatmap = (canonical_heatmap - np.nanmin(canonical_heatmap)) / (np.nanmax(canonical_heatmap) - np.nanmin(canonical_heatmap))

            axs[riga, colonna].imshow(face)
            axs[riga, colonna].imshow(canonical_heatmap, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_canonical.npy'), canonical_heatmap)

        if tot_Anger == 0 and emotion != 'ANGRY':
            axs[riga, 0].imshow(face)
            
        elif emotion != 'ANGRY':
            CSM_Anger = CSM_Anger / tot_Anger
            CSM_Anger = (CSM_Anger - np.nanmin(CSM_Anger)) / (np.nanmax(CSM_Anger) - np.nanmin(CSM_Anger))
            if ext == True:
                CSM_Anger = CSM_Anger/np.nanmax(np.abs(CSM_Anger))
                if solo_pos:
                    CSM_Anger[CSM_Anger < 0] = 0
            axs[riga, 0].imshow(face)
            axs[riga, 0].imshow(CSM_Anger, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Anger.npy'), CSM_Anger)

        if tot_Disgust == 0 and emotion != 'DISGUST':
            axs[riga, 1].imshow(face)
            
        elif emotion != 'DISGUST':
            CSM_Disgust = CSM_Disgust / tot_Disgust
            if ext == True:
                CSM_Disgust = CSM_Disgust/np.nanmax(np.abs(CSM_Disgust))
                if solo_pos:
                    CSM_Disgust[CSM_Disgust < 0] = 0
            else:
                CSM_Disgust = (CSM_Disgust - np.nanmin(CSM_Disgust)) / (np.nanmax(CSM_Disgust) - np.nanmin(CSM_Disgust))
            axs[riga, 1].imshow(face)
            axs[riga, 1].imshow(CSM_Disgust, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Disgust.npy'), CSM_Disgust)

        if tot_Fear == 0 and emotion != 'FEAR':
            axs[riga, 2].imshow(face)
            
        elif emotion != 'FEAR':
            CSM_Fear = CSM_Fear / tot_Fear
            
            if ext == True:
                CSM_Fear = CSM_Fear/np.nanmax(np.abs(CSM_Fear))
                if solo_pos:
                    CSM_Fear[CSM_Fear < 0] = 0
            else:
                CSM_Fear = (CSM_Fear - np.nanmin(CSM_Fear)) / (np.nanmax(CSM_Fear) - np.nanmin(CSM_Fear))
            axs[riga, 2].imshow(face)
            axs[riga, 2].imshow(CSM_Fear, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Fear.npy'), CSM_Fear)

        if tot_Happiness == 0 and emotion != 'HAPPY':
            axs[riga, 3].imshow(face)

        elif emotion != 'HAPPY':
            CSM_Happiness = CSM_Happiness / tot_Happiness
            
            if ext == True:
                CSM_Happiness = CSM_Happiness/np.nanmax(np.abs(CSM_Happiness))
                if solo_pos:
                    CSM_Happiness[CSM_Happiness < 0] = 0
            else:
                CSM_Happiness = (CSM_Happiness - np.nanmin(CSM_Happiness)) / (np.nanmax(CSM_Happiness) - np.nanmin(CSM_Happiness))
            axs[riga, 3].imshow(face)
            axs[riga, 3].imshow(CSM_Happiness, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Happiness.npy'), CSM_Happiness)

        if tot_Neutral == 0 and emotion != 'NEUTRAL':
            axs[riga, 4].imshow(face)
            
        elif emotion != 'NEUTRAL':
            CSM_Neutral = CSM_Neutral / tot_Neutral
            
            if ext == True:
                CSM_Neutral = CSM_Neutral/np.nanmax(np.abs(CSM_Neutral))
                if solo_pos:
                    CSM_Neutral[CSM_Neutral < 0] = 0
            else:
                CSM_Neutral = (CSM_Neutral - np.nanmin(CSM_Neutral)) / (np.nanmax(CSM_Neutral) - np.nanmin(CSM_Neutral))
            axs[riga, 4].imshow(face)
            axs[riga, 4].imshow(CSM_Neutral, alpha=al, cmap=colore,   vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Neutral.npy'), CSM_Neutral)

        if tot_Sadness == 0 and emotion != 'SAD':
            axs[riga, 5].imshow(face)
            
        elif emotion != 'SAD':
            CSM_Sadness = CSM_Sadness / tot_Sadness
            
            if ext == True:
                CSM_Sadness = CSM_Sadness/np.nanmax(np.abs(CSM_Sadness))
                if solo_pos:
                    CSM_Sadness[CSM_Sadness < 0] = 0
            else:
                CSM_Sadness = (CSM_Sadness - np.nanmin(CSM_Sadness)) / (np.nanmax(CSM_Sadness) - np.nanmin(CSM_Sadness))
            axs[riga, 5].imshow(face)
            axs[riga, 5].imshow(CSM_Sadness, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Sadness.npy'), CSM_Sadness)

        if tot_Surprise == 0 and emotion != 'SURPRISE':
            axs[riga, 6].imshow(face)
            
        elif emotion != 'SURPRISE':
            CSM_Surprise = CSM_Surprise / tot_Surprise
            
            if ext == True:
                CSM_Surprise = CSM_Surprise/np.nanmax(np.abs(CSM_Surprise))
                if solo_pos:
                    CSM_Surprise[CSM_Surprise < 0] = 0
            else:
                CSM_Surprise = (CSM_Surprise - np.nanmin(CSM_Surprise)) / (np.nanmax(CSM_Surprise) - np.nanmin(CSM_Surprise))
            axs[riga, 6].imshow(face)
            axs[riga, 6].imshow(CSM_Surprise, alpha=al, cmap=colore, vmin=minimo, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Surprise.npy'), CSM_Surprise)
        
        riga += 1
        colonna += 1

    ############################################ SALVARE IMMAGINE ############################################
    if save_img:
        plt.savefig(os.path.join(results_path, f'CSM.png'))
    if visualize:
        plt.show()
   
 

# minimo = 0   # da mettere -1 per le extremal quando non si vuole fare solo i positivi
# ext = False # da mettere true per le extremal
# solo_pos = False  # da mettere true per le extremal quando si vuole fare solo i positivi
# colore = 'jet'  # se si mette come min -1 mettere 'PiYG' altrimenti 'jet'
# output_path = f"Matrici_GradCam_Layer8_corretto" # path dove c'è la cartella heatmaps e il file results.csv
# limite = 1
# ############################################ SALVARE HEATMAP ############################################
# results_path = 'HEATMAPS/YOLO_Gradcam8'  # se si vogliono salvare le heatmap mettere il path desiderato
# # crea la cartella se non esiste
# if not os.path.exists(results_path):
#     os.makedirs(results_path)
# save_heatmap = True # da mettere true se si vogliono salvare le heatmap
# visualize = False  # da mettere true se si vuole visualizzare l'immagine
# ############################################ SALVARE immagine ############################################
# save_img = False # da mettere true se si vuole salvare l'immagine (la confusion matrix)
# al = 0.6 # alpha della heatmap (quanto è trasparente), si può non cambiare
# genearte_csm(output_path, limite, minimo, ext, solo_pos, colore, results_path, save_heatmap, save_img, al, visualize)
# print("Extremal perturbations completed")

def process_model(model_name, input_saliencies_path, test_set_images_path, extract_image_index_function, 
                 results_csv_name="predictions_csmable.csv", models_evaluations_folders_dictionary=OCCFT_MODELS_RESULTS_PATHS):
    """
    Process a single model folder to generate Canonical Saliency Maps (CSM).
    """    

    # Ensure the output directory exists
    results_path = os.path.join(input_saliencies_path, "HEATMAPS")
    os.makedirs(results_path, exist_ok=True)

    # Check if the required results.csv file exists
    model_results_folder = models_evaluations_folders_dictionary[model_name]
    results_csv_path = os.path.join(model_results_folder, results_csv_name)
    if not os.path.exists(results_csv_path):
        print(f"[WARNING] Missing {results_csv_name} in {model_name}. Skipping...")
        return

    # Run the CSM generation script
    generate_csm(
        input_saliencies_path=input_saliencies_path,
        results_csv_path=results_csv_path,
        test_set_images_path=test_set_images_path,
        results_path=results_path,

        extract_image_index_function=extract_image_index_function,
        
        limite=1,
        minimo=0,
        ext=False,
        solo_pos=False,
        colore="jet",
        al=0.6,
        
        save_heatmap=SAVE_HEATMAP,
        save_img=SAVE_IMG,
        visualize=VISUALIZE,
    )
    print(f"Completed processing for model: {model_name}")
    

# def extract_image_index_occluded(filename):
#     # check that the npys generated by the XAI methods for each image of the test set have the same numbering as the filanme of the original image processed.

#     # like bosphorus_bs001_ANGRY_30__masked-negative-DISGUST_mismatch.png
#     if not filename.startswith("bosphorus"):
#         raise ValueError(f"Unexpected filename format: {filename}")
    
#     index_part = filename.split('_')[1]
#     # remove bs and convert to int
#     index = int(index_part.replace("bs", ""))
#     return index


# ==================================== ARGUMENT PARSING AND SETTINGS ====================================

parser = argparse.ArgumentParser(description="Generate bubble-based explanations for model predictions on a test set.")
parser.add_argument('--redirect_output',    action='store_true',                    help="Redirect console output to a log file.")
parser.add_argument('--run_name',           type=str,  required=True,               help="Name for the run.")
parser.add_argument('--input_base_folder',  type=str,  default=SAVED_IMAGES_PATH,   help="Base folder where saliency maps are stored.")
parser.add_argument('--xai_method',         type=str,  required=True,   choices=["extpert", "gradcam"], help="XAI method to use for generating explanations: either extpert or gradcam.")
args = parser.parse_args()

# RUN
SAL_MAPS_FOLDER = os.path.join(args.input_base_folder, args.run_name)

if 'ext' not in args.run_name and 'gradcam' not in args.run_name:
    raise ValueError("run_name must contain either 'ext' or 'gradcam' to specify the XAI method used.")

test_set_name = get_test_set_name_from_run_name(args.run_name)
TEST_SET_IMAGES_PATH = TEST_SET_PATHS[test_set_name]['unoccluded_back_images_path']
EXTRACT_IMAGE_INDEX_FUNCTION = extract_image_index_original_or_180rotated # change this if pattern of filenames changes

# Redirect output if specified
if args.redirect_output:
    LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}_DO_CSM_from_{args.run_name}.log")
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(LOG_FILE_PATH)
    sys.stderr = Tee(LOG_FILE_PATH)

SAVE_HEATMAP = True
SAVE_IMG = True
VISUALIZE = False

print(f"========== SETTINGS ==========")
print(f"ARGS:")
print(f"\t--redirect_output: {args.redirect_output}")
print(f"\t--input_base_folder: {args.input_base_folder}")   
print(f"\t--run_name: {args.run_name}")
print(f"MACROS:")
print(f"\tSAL_MAPS_FOLDER: {SAL_MAPS_FOLDER}")
if args.redirect_output:
    print(f"\tLOG_FILE_PATH: {LOG_FILE_PATH}")
print(f"\tSAVE_HEATMAP: {SAVE_HEATMAP}")
print(f"\tSAVE_IMG: {SAVE_IMG}")
print(f"\tVISUALIZE: {VISUALIZE}")
print(f"==============================")

# =================================================================================================================

# Example usage:
# >>> test run extpert
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_csm.py" --redirect_output --run_name 20260224-202418_cmplt-run_occft-models_occluded-testset_do_explainability_extpert_keras --xai_method extpert
# >>> test run gradcam
# & C:/Users/Dragos/.conda/envs/fer-thesis/python.exe "c:/Users/Dragos/Roba/Lectures/YM2.2/Thesis/e Models/scripts/xai/do_csm.py" --redirect_output --run_name 20260227-174619_quick-run_occft-models_occluded-testset_do_explainability_gradcam_keras --xai_method gradcam
def main():
    """
    Main function to process all models and generate Canonical Saliency Maps (CSM).
    """
    # Get the list of models in the saliency maps folder
    models_path = os.path.abspath(SAL_MAPS_FOLDER)
    if not os.path.exists(models_path):
        print(f"[ERROR] Saliency maps folder not found: {models_path}")
        return

    model_names = [model for model in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, model))]
    model_names = [model_name for model_name in model_names if "heatmaps" not in model_name.lower()]  # Exclude any folder named "heatmaps"
    if not model_names:
        print(f"[ERROR] No models found in {models_path}")
        return

    print(f"Found {len(model_names)} models to process.")

    # Process each model
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        # if we are doing gradcam then we need to loop through all the layer folders too
        if args.xai_method == "gradcam":
            model_saliencies_path = os.path.join(SAL_MAPS_FOLDER, model_name)
            layer_folders = [folder for folder in os.listdir(model_saliencies_path) if os.path.isdir(os.path.join(model_saliencies_path, folder))]
            for layer_folder in layer_folders:
                input_saliencies_path = os.path.join(SAL_MAPS_FOLDER, model_name, layer_folder)
                process_model(model_name, input_saliencies_path, TEST_SET_IMAGES_PATH, EXTRACT_IMAGE_INDEX_FUNCTION)
        elif args.xai_method == "extpert":
            input_saliencies_path = os.path.join(SAL_MAPS_FOLDER, model_name)
            process_model(model_name, input_saliencies_path, TEST_SET_IMAGES_PATH, EXTRACT_IMAGE_INDEX_FUNCTION)
        else:
            raise ValueError(f"Unsupported XAI method: {args.xai_method}")

    print("All models processed successfully.")

if __name__ == "__main__":
    main()
    
