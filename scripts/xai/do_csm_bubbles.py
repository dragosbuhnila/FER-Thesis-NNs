# cdm stands for Canonical Saliency Map. 
# Credits to Adele Roggia for basically the whole code. (Dragos Buhnila) I merely reorganized it a little and added argparsing.
import os; import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
from typing import Tuple, Union
from scipy.interpolate import griddata
from PIL import Image
from tqdm import tqdm
import argparse

from modules.config import ADELE_TEST_SET_IMAGES_PATH, ALL_MODELS_PATHS, CANONICAL_FACES_CUT, EMOTIONS, LANDMARKER_MODEL_PATH, OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH, SAVED_IMAGES_PATH

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


def remove_miniscule_values(heatmap, tot, eps):
    heatmap = heatmap if (tot > 0 and float(np.max(heatmap)) > eps) else np.zeros_like(heatmap, dtype=float)
    return heatmap


def sottrazione_norm(model_bubbles_path, img_name, emotion_gt, emotion_predicted, only_pos = False):
    # img è il nome dell'immagine senza estensione
    # emotion è la stringa che indica l'emozione predetta
    # path è il percorso in cui si trovano le immagini
    folder = os.path.join(model_bubbles_path, f"{emotion_gt}")
    img_list = os.listdir(folder)
    # devo trovare il nome dell'immagine che contenga f"{img_name}_normplane_{emotion_predicted}_numerosity"
    true_plane = None
    for image in img_list:
        if f"{img_name}_normplane_{emotion_predicted}_numerosity" in image:
            true_plane = Image.open(os.path.join(model_bubbles_path, emotion_gt, image))
            true_plane = true_plane.convert('L')
            # normalizza tra 0 e 1
            true_plane = np.array(true_plane) / 255
            break
    # controllare che true_plane non sia vuoto
    if true_plane is None:
        print(f"[WARNING] True plane is None for {img_name}, prediced emotion: {emotion_predicted}")
        return None
    false_plane = Image.open(os.path.join(model_bubbles_path, f"{emotion_gt}", f"{img_name}_normglobalfalseplane.png"))
    false_plane = false_plane.convert('L')
    # normalizza tra 0 e 1
    false_plane = np.array(false_plane) / 255
    # sottrazione
    diff = true_plane - false_plane
    # normalizzo tra -1 e 1
    
    if only_pos:
        diff[diff < 0] = 0
    diff = diff / np.max(np.abs(diff))

    return diff

def compute_csm_bubbles(model_bubbles_path, limite, results_path, positivi, lim_m, mappa_colore, save_image, save_heatmap, visualize):

    base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    dataset_images_path = OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH
    print(f'[WARNING] hardcoded path for images to occluded test set. You should parametrize this. Path: {dataset_images_path}')
    canonical_faces_path = CANONICAL_FACES_CUT  # path delle facce canoniche

    al = 0.6
    lim = limite

    # Anti-rumore overlay (EVITA "aloni blu" su celle che devono restare "base only")
    EPS = 1e-6
    
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
        bubbles_images_for_current_gt_emotion = os.listdir(os.path.join(model_bubbles_path, emotion))
        dataset_images = os.listdir(os.path.join(dataset_images_path, emotion))
        
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


        for idx, dataset_image in enumerate(tqdm(dataset_images, desc=f"Processing {emotion} images", unit="image")):
            # control the true class abd the predicted class
            # prendere l'elenco di immagini nella cartella emot_folder con il nome che inizia con img
            # my current names look like the following: image_50_3d841d89d769a83e3f6c458f3f06e921.png
            dataset_image_filename_full = dataset_image.split('.')[0]
            dataset_image_essential_name_d3 = dataset_image_filename_full.split('_')[0] + "_" + dataset_image_filename_full.split('_')[1] 
            dataset_image_essential_name_d0 = dataset_image_filename_full.split('_')[0] + "_" + str(int(dataset_image_filename_full.split('_')[1]))
            gt_emotion = emotion

            for filename in bubbles_images_for_current_gt_emotion:
                if f"{dataset_image_essential_name_d3}__predictedclass" in filename:
                    predicted_class = filename.split('_')[4]
                    dataset_image_essential_name = dataset_image_essential_name_d3
                    if predicted_class not in EMOTIONS:
                        raise ValueError(f"Predicted class {predicted_class} not in EMOTIONS list")       
                    break
                elif f"{dataset_image_essential_name_d0}__predictedclass" in filename:
                    predicted_class = filename.split('_')[4]
                    dataset_image_essential_name = dataset_image_essential_name_d0
                    if predicted_class not in EMOTIONS:
                        raise ValueError(f"Predicted class {predicted_class} not in EMOTIONS list")       
                    break
            

            heatmap = sottrazione_norm(model_bubbles_path, dataset_image_essential_name, gt_emotion, predicted_class, only_pos=positivi)

            if heatmap is None:
                continue

            image = mp.Image.create_from_file(os.path.join(dataset_images_path, emotion, dataset_image))
            Iwidth = image.width
            Iheight = image.height
            Idetection_result = detector.detect(image)
            
            image_landmarks_list = Idetection_result.face_landmarks
            # se il volto non è stato rilevato, segnala l'errore
            if image_landmarks_list == None:
                print(f"Face not detected in {dataset_image}")
                continue
            
            heatmap_values = []
            for idx in range(len(image_landmarks_list[0])):
                image_landmarks = image_landmarks_list[0][idx]
                if image_landmarks == None:
                    print(f"Landmark {idx} not detected in {dataset_image}")
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
            grid_z = grid_z.T
            # se grid_z è nan visualizza l'immagine
            if np.isnan(grid_z).all():
                print(f"NaN in {dataset_image}")
                continue
            
            num = 1

            if emotion != predicted_class:
                tot_w += num
                canonical_heatmap_w = canonical_heatmap_w + grid_z*num
                if predicted_class == 'ANGRY':
                    CSM_Anger = CSM_Anger + grid_z*num
                    tot_Anger += num
                elif predicted_class == 'DISGUST':
                    CSM_Disgust = CSM_Disgust + grid_z*num
                    tot_Disgust += num
                elif predicted_class == 'FEAR':
                    CSM_Fear = CSM_Fear + grid_z*num
                    tot_Fear += num
                elif predicted_class == 'HAPPY':
                    CSM_Happiness = CSM_Happiness + grid_z*num
                    tot_Happiness += num
                elif predicted_class == 'NEUTRAL':
                    CSM_Neutral = CSM_Neutral + grid_z*num
                    tot_Neutral += num
                elif predicted_class == 'SAD':
                    CSM_Sadness = CSM_Sadness + grid_z*num
                    tot_Sadness += num
                elif predicted_class == 'SURPRISE':
                    CSM_Surprise = CSM_Surprise + grid_z*num
                    tot_Surprise += num
            else:
                tot += num
                canonical_heatmap = canonical_heatmap + grid_z*num
            

        
        
        face = cv2.imread(os.path.join(canonical_faces_path, f"{emotion}.png"))

        if tot == 0:
            axs[riga, colonna].imshow(face)
            #axs[riga, colonna].set_title(f'{emotion}-{emotion}')
            axs[riga, colonna].axis('off')
        else:
            canonical_heatmap = canonical_heatmap / tot
            canonical_heatmap = (canonical_heatmap - np.nanmin(canonical_heatmap)) / (np.nanmax(canonical_heatmap) - np.nanmin(canonical_heatmap))
            axs[riga, colonna].imshow(face)
            axs[riga, colonna].imshow(canonical_heatmap, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_canonical.npy'), canonical_heatmap)

        if tot_Anger == 0 and emotion != 'ANGRY':
            axs[riga, 0].imshow(face)
            #axs[riga, 0].set_title(f'Anger-{emotion}')
            #axs[riga, 0].axis('off')
        elif emotion != 'ANGRY':
            CSM_Anger = CSM_Anger / tot_Anger
            CSM_Anger = (CSM_Anger - np.nanmin(CSM_Anger)) / (np.nanmax(CSM_Anger) - np.nanmin(CSM_Anger))
            axs[riga, 0].imshow(face)
            axs[riga, 0].imshow(CSM_Anger, alpha=al,cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Anger.npy'), CSM_Anger)

        if tot_Disgust == 0 and emotion != 'DISGUST':
            axs[riga, 1].imshow(face)
        elif emotion != 'DISGUST':
            CSM_Disgust = CSM_Disgust / tot_Disgust
            CSM_Disgust = (CSM_Disgust - np.nanmin(CSM_Disgust)) / (np.nanmax(CSM_Disgust) - np.nanmin(CSM_Disgust))
            axs[riga, 1].imshow(face)
            axs[riga, 1].imshow(CSM_Disgust, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Disgust.npy'), CSM_Disgust)

        if tot_Fear == 0 and emotion != 'FEAR':
            axs[riga, 2].imshow(face)
            
        elif emotion != 'FEAR':
            CSM_Fear = CSM_Fear / tot_Fear
            CSM_Fear = (CSM_Fear - np.nanmin(CSM_Fear)) / (np.nanmax(CSM_Fear) - np.nanmin(CSM_Fear))
            axs[riga, 2].imshow(face)
            axs[riga, 2].imshow(CSM_Fear, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Fear.npy'), CSM_Fear)

        if tot_Happiness == 0 and emotion != 'HAPPY':
            axs[riga, 3].imshow(face)
            
        elif emotion != 'HAPPY':
            CSM_Happiness = CSM_Happiness / tot_Happiness
            CSM_Happiness = (CSM_Happiness - np.nanmin(CSM_Happiness)) / (np.nanmax(CSM_Happiness) - np.nanmin(CSM_Happiness))
            axs[riga, 3].imshow(face)
            axs[riga, 3].imshow(CSM_Happiness, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Happiness.npy'), CSM_Happiness)

        if tot_Neutral == 0 and emotion != 'NEUTRAL':
            axs[riga, 4].imshow(face)
            
        elif emotion != 'NEUTRAL':
            CSM_Neutral = CSM_Neutral / tot_Neutral
            CSM_Neutral = (CSM_Neutral - np.nanmin(CSM_Neutral)) / (np.nanmax(CSM_Neutral) - np.nanmin(CSM_Neutral))
            axs[riga, 4].imshow(face)
            axs[riga, 4].imshow(CSM_Neutral, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Neutral.npy'), CSM_Neutral)

        if tot_Sadness == 0 and emotion != 'SAD':
            axs[riga, 5].imshow(face)
            
        elif emotion != 'SAD':
            CSM_Sadness = CSM_Sadness / tot_Sadness
            CSM_Sadness = (CSM_Sadness - np.nanmin(CSM_Sadness)) / (np.nanmax(CSM_Sadness) - np.nanmin(CSM_Sadness))
            axs[riga, 5].imshow(face)
            axs[riga, 5].imshow(CSM_Sadness, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path,f'{emotion}_Sadness.npy'), CSM_Sadness)

        if tot_Surprise == 0 and emotion != 'SURPRISE':
            axs[riga, 6].imshow(face)
            
        elif emotion != 'SURPRISE':
            CSM_Surprise = CSM_Surprise / tot_Surprise
            CSM_Surprise = (CSM_Surprise - np.nanmin(CSM_Surprise)) / (np.nanmax(CSM_Surprise) - np.nanmin(CSM_Surprise))
            axs[riga, 6].imshow(face)
            axs[riga, 6].imshow(CSM_Surprise, alpha=al, cmap=mappa_colore, vmin=lim_m, vmax=lim)
            if save_heatmap:
                np.save(os.path.join(results_path, f'{emotion}_Surprise.npy'), CSM_Surprise)
        
        riga += 1
        colonna += 1

    if save_image:
        plt.savefig(os.path.join(results_path, f'CSM_sottrazione_norm.png'))
    if visualize:
        plt.show()
   

def compute_csm_bubbles_wrapper(model_bubbles_path, bubbles_path, positivi,
                                save_image=True, save_heatmap=True, visualize=False,
                                mappa_colore = 'jet'):   # PiYG (con -1 come limite inf) o jet
    model_name = os.path.basename(model_bubbles_path)
    results_final_dir_name = f"Bubbles_CSM_{model_name}"
    results_path = os.path.join(bubbles_path, 'HEATMAPS', results_final_dir_name) # path dove salvare i risultati 
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    limite = 1 
    if positivi:
        lim_m = 0 # mettere -1 quando positivi è false
    else:
        lim_m = -1

    compute_csm_bubbles(model_bubbles_path, limite, results_path, positivi, lim_m, mappa_colore, save_image, save_heatmap, visualize)
    print(f"completed")

# ================================= SETTINGS =================================

argparser = argparse.ArgumentParser(description='Compute CSM bubbles for all models.')
argparser.add_argument('--run_name', type=str, default='20260221-184815_bubbles_cmplt-run_occft-models_original-testset', help='Name of the run to process. It should match the folder name in SAVED_IMAGES_PATH where the bubble images are stored.')
args = argparser.parse_args()



MODELS_NAMES = [model_name for model_name in ALL_MODELS_PATHS.keys() if 'occft' in model_name]
# TODO: check if this flag results in the wrong results, as I'm not sure what it does. I want to do the subtraction, why should i want to see just positives?
#   If this flag is true, it will set to 0 all the negative values of the heatmap, so it will show only the positive values. If it's false, it will show both positive and negative values.
#   The original code set it to true
POSITIVI = True # se si vogliono vedere solo i positivi 
RESULTS_NAME_SIGNATURE = "HEATMAPS"

print(f"======================= SETTINGS =======================")
print(f"MACROS:")
print(f"\tPOSITIVI: {POSITIVI}")
print(f"\tMODELS_NAMES: {MODELS_NAMES}")
print(f"\tRESULTS_NAME_SIGNATURE: {RESULTS_NAME_SIGNATURE}")
print(f"========================================================")

if __name__ == "__main__":
    bubbles_path = os.path.join(SAVED_IMAGES_PATH, args.run_name)
    folders = [f for f in os.listdir(bubbles_path) if os.path.isdir(os.path.join(bubbles_path, f))]
    folders = [folder for folder in folders if RESULTS_NAME_SIGNATURE not in folder]
    for folder in folders:
        model_bubbles_path = os.path.join(bubbles_path, folder)
        print(f"Processing folder: {folder}")
        print(f"\tmodel_bubbles_path: {model_bubbles_path}")
        # print(f"\tbubbles_path: {bubbles_path}")
        # print(f"\tPOSITIVI: {POSITIVI}")
        compute_csm_bubbles_wrapper(model_bubbles_path, bubbles_path, positivi=POSITIVI)
