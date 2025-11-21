# This module comes from the other part of the project which focused on occlusion masking and saliency map analysis.
# The original module is called "landmark_utils.py".
# This right file may be updated if needed, so don't necessarily treat it as the same as the original one.
# Dragos.

from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from PIL import Image


from modules.config import FORCE_RECALCULATE_LANDMARKS, LANDMARK_COORDINATES_FOLDER, LANDMARKER_MODEL_PATH, GLOBALS
from modules.misc import hash_image

EMOTION_AUS = {
    "ANGRY":    ["AU4", "AU23", "AU24"],
    "DISGUST":  ["AU9", "AU10", "AU16"],
    "FEAR":     ["AU1+2", "AU27"],
    "HAPPY":    ["AU6", "AU12"],
    "SAD":      ["AU1", "AU15"],
    "SURPRISE": ["AU1+2+5", "AU26"],
}

AU_SUBSET = { # 14 AUs
    "AU1" :  "AU1",
    "AU2" :  "AU2",
    "AU4" :  "AU4",
    "AU5" :  "AU5",
    "AU6" :  "AU6",
    "AU9" :  "AU9",
    "AU10": "AU10",
    "AU12": "AU12",
    "AU15": "AU15",  
    "AU16": "AU16",
    "AU23": "AU23", 
    "AU24": "AU24", 
    "AU26": "AU26", 
    "AU27": "AU27",
}

class LandmarkDict(dict):
    def __getitem__(self, key):
        # Custom logic before returning the value
        key = AU_SUBSET[key] if key in AU_SUBSET else key
        value = super().__getitem__(key)
        return value

AU_LANDMARKS = LandmarkDict({
    # ========================================================================
    "AU1+2+5": [[46, 105, 66, 107,                              # Eyebrows Upper L
                190, 56, 28, 27, 29, 247,                       # Eye Upper L
                46],            
                [336, 296, 334, 283,                            # Eyebrows Upper R
                467, 259, 257, 258, 286, 414,                   # Eye Upper R
                336]],

    "AU1+2":   [[46, 105, 66, 107,                              # Eyebrows Upper L
                46],            
                [336, 296, 334, 283,                            # Eyebrows Upper R
                336]],

    # AU1 === inner brow raiser === frontalis muscle (the medial portion) === 
    "AU1":      [[66, 107], [336, 296]],                        # Eyebrows Upper Internal

    # "AU1":      [66, 107, 336, 296],                          # Eyebrows Upper Internal

    # "AU1full":  [52, 65, 55, 285, 295, 282,                   # Eyebrow Lower     
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             69, 108, 337, 299],                           # some medial frontalis

    # AU2 === outer brow raiser === frontalis muscle (the lateral portion) === 
    "AU2":      [[46, 105], [334, 283]],                        # Eyebrows Upper Outer (contains some lower)

    # "AU2":      [46, 105, 334, 283],                          # Eyebrows Upper Outer (contains some lower)

    # "AU2full":  [52, 65, 55, 285, 295, 282,                   # Eyebrow Lower     
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             104, 333],                                    # some lateral frontalis

    # AU4 === brow lowerer ==== corrugator supercilii, depressor supercilii, and/or procerus muscles === 
    "AU4":      [[46, 105, 66, 107, 9, 336, 296, 334, 283]],    # Eyebrows Upper (contains some lower)

    # "AU4":      [46, 105, 66, 107, 9, 336, 296, 334, 283],    # Eyebrows Upper (contains some lower)

    # "AU4full":  [52, 65, 55, 8, 285, 295, 282,                # Eyebrow Lower 
    #             46, 105, 66, 107, 9, 336, 296, 334, 283,      # Eyebrows Upper (contains some lower)
    #             8, 9],                                        # Center

    # AU5 === upper lid raiser === levator palpebrae superioris ===
    "AU5":      [[414, 286, 258, 257, 259, 467],                # Eye Upper R
                [247, 29, 27, 28, 56, 190]],                    # Eye Upper L

    # "AU5":      [414, 286, 258, 257, 259, 467,                # Eye Upper R
                #  247, 29, 27, 28, 56, 190],                   # Eye Upper L

    # "AU5full":  [414, 286, 258, 257, 259, 467,                # Eye Upper R
    #             247, 29, 27, 28, 56, 190,                     # Eye Upper L
    #             256, 253, 339, 446,                           # Eye Lower R
    #             226, 110, 23, 26, 243],                       # Eye Lower L  

    # AU6 === cheek raiser === orbicularis oculi muscle (the orbital portion) ===
    "AU6":      [[256, 253, 339, 446,                           # Eye Lower R
                346, 329, 277,                                  # Cheek Upper R
                256],
                [226, 110, 23, 26, 243,                         # Eye Lower L  
                 47, 100, 117,                                  # Cheek Upper L
                 226]],

    # "AU6":      [256, 253, 339, 446,                          # Eye Lower R
    #             226, 110, 23, 26, 243,                        # Eye Lower L  
    #             277, 329, 348, 347, 346,                      # Cheek Upper R
    #             117, 118, 119, 100, 47],                      # Cheek Upper L

    # "AU6full":  [414, 286, 258, 257, 259, 467,                # Eye Upper R
    #             247, 29, 27, 28, 56, 190,                     # Eye Upper L
    #             256, 253, 339, 446,                           # Eye Lower R
    #             226, 110, 23, 26, 243,                        # Eye Lower L  
    #             277, 329, 348, 347, 346,                      # Cheek Upper R
    #             117, 118, 119, 100, 47],                      # Cheek Upper L

    # AU9 === nose wrinkler === levator labii superioris alaeque nasi === 
    "AU9":      [[195,
                 355, 279, 358, 392,                 # Nose Right (added 358 for gap in inverse mask)
                 166, 129, 49, 126,                   # Nose Left  (added 129 for gap in inverse mask)
                 195]],

    # "AU9":      [417, 351, 399, 420, 279, 392,                # Nose Right
    #             193, 122, 174, 198, 49, 166],                 # Nose Left  

    # "AU9full": [417, 351, 399, 420, 279, 392,                 # Nose Right
    #             193, 122, 174, 198, 49, 166,                  # Nose Left
    #             1,],                                          # Nose Center

    # AU10 === upper lip raiser === levator labii superioris === 
    "AU10":     [[216, 206, 203, 423, 426, 436,                 # NasoLabial Fold
                  291, 269, 0, 39,61,                           # Upper Lip Outer
                 216]],

    # "AU10":     [216, 206, 203, 423, 426, 436,                # NasoLabial Fold
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,],                                # Upper Lip Inner

    # "AU10full": [329, 429,                                    # levator labii superioris Right 
    #             100, 209,                                     # levator labii superioris Left  
    #             216, 206, 423, 203, 426, 436,                 # NasoLabial Fold
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,],                                # Upper Lip Inner

    # AU12 === lip corner puller === zygomaticus major ===
    "AU12":     [[436, 422,                                     # Mouth Side R
                405, 15, 181,                                   # Lower Lip Outer Alt w/o Chelions
                202, 216,                                       # Mouth Side L 
                436]],

    # "AU12":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             202, 216, 436, 422],                          # Mouth Side
    
    # "AU12full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             202, 216, 203, 423, 436, 422],                # Mouth Side

    # AU15 === lip corner depressor === depressor anguli oris ===
    "AU15":     [[39, 0, 269,                                   # Upper Lip Outer w/o Chelions
                287,                                            # Lip corner extension R
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                57,                                             # Lip corner extension L
                39]],

    # "AU15":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 287],                                     # Lip corner extension

    # "AU15full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 287],                                     # Lip corner extension

    # AU16 === lower lip depressor === depressor labii inferioris === 
    "AU16":     [[291, 404, 16, 180, 61]],                      # Lower Lip Mid

    # "AU16":     [88, 14, 318,                                 # Lower Lip Inner
    #             61, 181, 17, 405, 291],                       # Lower Lip Outer

    # "AU16full": [88, 14, 318,                                 # Lower Lip Inner
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU23 === lip tightener === orbicularis oris muscle (marginal portion) === 
    "AU23":     [[39, 0, 269,                                   # Upper Lip Outer w/o Chelions
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                39]],

    # "AU23":     [39, 0, 269,                                  # Upper Lip Outer w/o Chelions
    #              88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #              181, 17, 405],                               # Lower Lip Outer w/o Chelions

    # "AU23full": [57, 165, 164, 391, 287,                      # Above Lips
    #             39, 0, 269,                                   # Upper Lip Outer w/o Chelions
    #             81, 13, 311,                                  # Upper Lip Inner w/o Chelions
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             181, 17, 405,                                 # Lower Lip Outer w/o Chelions
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU24 === lip presser === orbicularis oris muscle (marginal portion) === 
    "AU24":     [[61, 39, 0, 269, 291,                          # Upper Lip Outer
                405, 17, 181,                                   # Lower Lip Outer w/o Chelions
                61]],

    # "AU24":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #              88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #              61, 181, 17, 405, 291,],                     # Lower Lip Outer

    # "AU24full": [57, 165, 164, 391, 287,                      # Above Lips
    #             61, 39, 0, 269, 291,                          # Upper Lip Outer
    #             81, 13, 311,                                  # Upper Lip Inner w/o Chelions
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             57, 182, 18, 406, 287],                       # Below Lips

    # AU26 === jaw drop === relaxation of masseter, temporalis, and medial pterygoid
    "AU26":     [[61, 180, 16, 404, 291,                        # Lower Lip Mid
                262, 428, 199, 208, 32,                         # Chin
                61]],

    # "AU26":     [88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # "AU26full": [88, 14, 318,                                 # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # AU27 === mouth stretch === lateral pterygoid and the suprahyoid (anterior digastric, geniohyoid, and mylohyoid) ===
    "AU27":     [[61, 39, 0, 269, 291,                          # Upper Lip Outer
                262, 428, 199, 208, 32,                         # Chin
                61]],

    # "AU27":     [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin

    # "AU27full": [61, 39, 0, 269, 291,                         # Upper Lip Outer
    #             88, 14, 318,                                  # Lower Lip Inner w/o Chelions
    #             61, 181, 17, 405, 291,                        # Lower Lip Outer
    #             32, 176, 148, 152, 377, 400, 262],            # Chin
})

FACE_PARTS_LANDMARKS = LandmarkDict({
    "Left Eyebrow": [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 
                      70]],
    "Right Eyebrow": [[336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 
                       336]],
    "Left Eye": [[35, 247, 470, 190, 244, 233, 23, 110, 35,
                  35]],
    "Right Eye": [[265, 467, 475, 414, 464, 453, 253, 339,
                   265]],
    # "Left Eye": [[33, 160, 158, 157, 173, 154, 145, 163
    #               , 33]],
    # "Right Eye": [[463, 384, 386, 388, 263, 390, 374, 381, 
    #               463]],
    "Left Cheek": [[117, 119, 114, 100, 142, 203, 206, 205, 50,
                    117]],
    "Right Cheek": [[346, 348, 343, 329, 371, 423, 426, 425, 280, 
                     346]],
    "Nose": [[168, 456, 360, 289, 19, 59, 131, 236,
              168]],  
    "Mouth": [[57, 186, 165, 164, 391, 410, 287, 
               422, 424, 418, 421, 200, 201, 194, 204, 202,
               57]],
    "Corrugator": [[8, 337, 108,
                    8]],
})

ROI_ORDER_FACEPARTS = ['Left Eyebrow', 'Right Eyebrow', 'Left Eye', 'Right Eye', 'Left Cheek', 'Right Cheek', 'Nose', 'Mouth', 'Corrugator']

FACE_PARTS_LANDMARKS_LRMERGED = LandmarkDict({
    "Eyebrows":     [FACE_PARTS_LANDMARKS["Left Eyebrow"][0], 
                     FACE_PARTS_LANDMARKS["Right Eyebrow"][0]],
    "Eyes":         [FACE_PARTS_LANDMARKS["Left Eye"][0], 
                     FACE_PARTS_LANDMARKS["Right Eye"][0]],
    "Cheeks":       [FACE_PARTS_LANDMARKS["Left Cheek"][0], 
                     FACE_PARTS_LANDMARKS["Right Cheek"][0]],
    "Nose":         [FACE_PARTS_LANDMARKS["Nose"][0]],
    "Mouth":        [FACE_PARTS_LANDMARKS["Mouth"][0]],
    "Corrugator":   [FACE_PARTS_LANDMARKS["Corrugator"][0]],
})

def detect_facial_landmarks_frompath(image_path):
    """
    Returns the coordinates for landmarks on an image.
    """
    # TODO: check if coordinates aren't already cached to save time (so you also need to add a feature to clear the cache in case you change something in the related MACROS)

    # Load the image
    image = Image.open(image_path)
    image_hash = hash_image(image)

    detect_facial_landmarks(np.asarray(image, dtype=np.uint8), image_hash)
        
def detect_facial_landmarks(numpy_image, image_hash, ignore_error, force_recalculate=FORCE_RECALCULATE_LANDMARKS):
    desired_landmarks = load_landmark_coordinates(image_hash)
    if desired_landmarks is not None and not force_recalculate:
        GLOBALS["TOTAL_IMAGES_LANDMARKS_LOADED"] += 1
        return np.array(desired_landmarks)

    # check if image is numpy
    if not isinstance(numpy_image, np.ndarray):
        raise TypeError("Input image must be a numpy array, or alternatively, a tf.Tensor which will be converted promptly.")

    # Convert a NumPy array or other image format to mediapipe.Image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # Set up FaceLandmarker
    model_path = LANDMARKER_MODEL_PATH
    base_options = BaseOptions(model_asset_path=model_path)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

        if result.face_landmarks:
            if len(result.face_landmarks) > 1:
                raise ValueError(f"Multiple faces detected. Only one face is expected.")
            face_landmarks = result.face_landmarks[0]
            if len(face_landmarks) < 4:
                raise ValueError(f"Insufficient landmarks detected, detected {len(face_landmarks)}.")
            
            desired_landmarks = []

            h, w, _ = numpy_image.shape
            for face_landmarks in result.face_landmarks:
                for idx in range(len(face_landmarks)):
                    landmark = face_landmarks[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    desired_landmarks.append((x, y))
            
            ok = save_landmark_coordinates(image_hash, desired_landmarks, force_recalculate)
            if ok == False:
                raise ValueError(f"Could not save landmark coordinates for image hash {image_hash}.")
            else: 
                GLOBALS["TOTAL_IMAGES_LANDMARKS_SAVED"] += 1

            return np.array(desired_landmarks)
        else:
            GLOBALS["UNLANDMARKABLE_IMAGES_LIST"].append(image_hash)
            if ignore_error:
                return np.array([])
            raise ValueError(f"No face detected in image with hash {image_hash}.")

def detect_facial_landmarks__batch(images, image_hashes, parallelize):
    if len(images) != len(image_hashes):
        raise ValueError("Mismatched lengths between images and image hashes.")

    if parallelize:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(detect_facial_landmarks, images, image_hashes, [True]*len(images)))
    else:
        # I don't need this to be tensorable anymore as I don't use it online anymore.
        # I just run it as overhead when generating the three data generators
        results = [res for res in map(detect_facial_landmarks, images, image_hashes, [True]*len(images))]
    return np.array(results)

def get_landmark_coordinate_sets_by_emotion(landmark_coordinates, emotion):
    """
    Given the full set of landmark coordinates and an emotion, returns the coordinates relevant to that emotion, as a list of lists of coordinates.

    :param landmark_coordinates: List of (x, y) tuples for all landmarks.
    :param emotion: The emotion string.
    :return: List of lists of (x, y) tuples corresponding to the landmarks for the specified AU.

    Example:
    If AU_LANDMARKS["AU1"] = [[66, 107], [336, 296]]
    and landmark_coordinates = [(x0, y0), (x1, y1), ..., (x467, y467)]
    then get_landmark_coordinate_sets_by_au(landmark_coordinates, "AU1") will return:
    [ [(x66, y66), (x107, y107)], [(x336, y336), (x296, y296)] ]
    """
    if len(landmark_coordinates) == 0:
        raise ValueError("landmark_coordinates is an empty list")
    if emotion == "NEUTRAL":
        # No AUs for neutral expression
        return []
    if emotion not in EMOTION_AUS:
        # Don't move this above if emotion == "NEUTRAL", bc there's no "NEUTRAL" in EMOTION_AUS, so it would error
        raise ValueError(f"Emotion {emotion} not found in EMOTION_AUS dictionary.")
    
    AUs = EMOTION_AUS[emotion]
    for au in AUs:
        if au not in AU_LANDMARKS:
            raise ValueError(f"AU {au} not found in AU_LANDMARKS dictionary.")
    
    au_landmarks_indices_sets = []
    for au in AUs:
        au_landmarks_indices_list_of_lists = AU_LANDMARKS[au]
        for au_landmarks_indices_list in au_landmarks_indices_list_of_lists:
            au_landmarks_indices_sets.append(au_landmarks_indices_list)  # list of lists of indices

    au_landmark_coords_sets = []

    for indices_set in au_landmarks_indices_sets:
        coords = [landmark_coordinates[idx] for idx in indices_set] # list of coordinates
        au_landmark_coords_sets.append(coords) # put this list of coordinates in a list, so that au_landmark_coords_sets becomes a list of lists of coordinates

    return au_landmark_coords_sets

def get_landmark_coordinate_sets_by_emotion__batch(landmark_coordinates_batch, emotions_batch, parallelize):
    if not isinstance(landmark_coordinates_batch, (list, tuple, np.ndarray)):
        raise TypeError("Input must be a list or tuple or np.ndarray of numpy arrays.")
    if not isinstance(emotions_batch, (list, tuple)):
        raise TypeError("Emotions must be a list or tuple of strings.")

    if len(landmark_coordinates_batch) != len(emotions_batch):
        raise ValueError("Mismatched lengths between landmark coordinates and emotions.")

    if parallelize:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(get_landmark_coordinate_sets_by_emotion, landmark_coordinates_batch, emotions_batch))
    else:
        # TODO: ensure it's tensorable
        results = [res for res in map(get_landmark_coordinate_sets_by_emotion, landmark_coordinates_batch, emotions_batch)]
    return results

def save_landmark_coordinates(image_hash, landmark_coordinates, force_recalculate, coords_folder=LANDMARK_COORDINATES_FOLDER):
    """
        Saves the landmark coordinates to a .npy file in the LANDMARK_COORDINATES_FOLDER.
        The path/filename includes emotion as folder and same name as the image
    """
    filename = f"{image_hash}.npy"  # Save as .npy

    if filename in os.listdir(coords_folder) and not force_recalculate:
        raise ValueError(f"Landmark coordinates for image hash {image_hash} already exist, meanining there has been a hash collision.")

    filepath = os.path.join(coords_folder, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure all subfolders exist
    np.save(filepath, np.array(landmark_coordinates, dtype=np.int32))


def load_landmark_coordinates(image_hash, coords_folder=LANDMARK_COORDINATES_FOLDER):
    """
        The path/filename includes emotion as folder and same name as the image
    """
    filename = f"{image_hash}.npy"
    filepath = os.path.join(coords_folder, filename)
    if not os.path.exists(filepath):
        return None
    return np.load(filepath)

def get_all_AUs():
    """
    Returns a list of all AUs available in the LANDMARKS dictionary.
    """
    basic_aus = [key for key in AU_LANDMARKS.keys() if not "+" in key]
    return basic_aus

def get_all_face_parts():
    """
    Returns a list of all face parts available in the FACE_PARTS_LANDMARKS dictionary.
    """
    return list(FACE_PARTS_LANDMARKS.keys())

def get_all_face_parts_lrmerged():
    """
    Returns a list of all face parts available in the FACE_PARTS_LANDMARKS dictionary.
    """
    return list(FACE_PARTS_LANDMARKS_LRMERGED.keys())