import os

# 0) Debug Options
DEBUG_MASKING = False                                                   # If True, will plot images during masking for debugging
DEBUG_MASKING_TIER2 = False if DEBUG_MASKING == False else True         # If True, will plot more images during masking for debugging

FORCE_RECALCULATE_LANDMARKS = False                                     # If True, will recalculate and overwrite existing landmark coordinates even if they exist

GLOBALS = {
    "TOTAL_IMAGES_LANDMARKS_SAVED": 0,                      # Counter for total images saved during masking
    "TOTAL_IMAGES_LANDMARKS_LOADED": 0,                     # Counter for total images loaded during masking
    "UNLANDMARKABLE_IMAGES_LIST": [],
}

# 1) Core Configurations
# ______________________________________________________________________
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# ______________________________________________________________________
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "data")
AUXILIARY_DATA_DIR = os.path.join(DATA_BASE_DIR, "auxiliary")

# 1a) Emotions
# ______________________________________________________________________
# Emotion labels MUST be in ALPHABETICAL ORDER for them to match to the dataset labels
EMOTIONS = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]
EMOTIONS_PRED = {
    "ANGRY": "Anger",
    "DISGUST": "Disgust",
    "FEAR": "Fear",
    "HAPPY": "Happiness",
    "NEUTRAL": "Neutral",
    "SAD": "Sadness",
    "SURPRISE": "Surprise"
}

# 1b) Masking
# ______________________________________________________________________
MASK_COLOR = (54, 61, 52) # Good ones: graphite_gray: (54, 61, 52)

# 2) Paths

# 2a) Datasets paths
# ______________________________________________________________________
DATASETS_PATH = os.path.join(DATA_BASE_DIR, "datasets")

# ______________________________________________________________________
# > ADELE_TEST_SET
ADELE_TEST_SET_BASE_PATH = os.path.join(DATASETS_PATH, "adele_test_set")
ADELE_TEST_SET_H5_PATH = os.path.join(ADELE_TEST_SET_BASE_PATH, "adele_test_set.h5")
ADELE_TEST_SET_IMAGES_PATH = os.path.join(ADELE_TEST_SET_BASE_PATH, "extracted_images")
ADELE_TEST_SET_YAML_PATH = os.path.join(ADELE_TEST_SET_BASE_PATH, "adele_test_set_yaml")
# ______________________________________________________________________
# > OCCLUDED_TEST_SET
OCCLUDED_TEST_SET_BASE_PATH = os.path.join(DATASETS_PATH, "occluded_test_set")
OCCLUDED_TEST_SET_H5_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "occluded_test_set.h5")
OCCLUDED_TEST_SET_IMAGES_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "bosphorus_test_HQ")
OCCLUDED_TEST_SET_RESIZED_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "output_images_testset_resized")
OCCLUDED_TEST_SET_YAML_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "occluded_test_set_yaml")
# ______________________________________________________________________
# > ORIGINAL_TRAIN_VAL_SET
ORIGINAL_TRAIN_VAL_SET_BASE_PATH = os.path.join(DATASETS_PATH, "original_train_val_set")
ORIGINAL_TRAIN_VAL_SET_H5_PATH = os.path.join(ORIGINAL_TRAIN_VAL_SET_BASE_PATH, "dataset.h5")
ORIGINAL_TRAIN_SET_IMAGES_PATH = os.path.join(ORIGINAL_TRAIN_VAL_SET_BASE_PATH, "dataset_extracted", "train")
ORIGINAL_VAL_SET_IMAGES_PATH = os.path.join(ORIGINAL_TRAIN_VAL_SET_BASE_PATH, "dataset_extracted", "val")
# ______________________________________________________________________
# > BOSPHORUS_TEST_HQ                   # this h5 will be full size not resized to small images
BOSPHORUS_TEST_HQ_BASE_PATH = os.path.join(DATASETS_PATH, "bosphorus_test_HQ")
BOSPHORUS_TEST_HQ_IMAGES_PATH = os.path.join(BOSPHORUS_TEST_HQ_BASE_PATH, "bosphorus_test_HQ")
BOSPHORUS_TEST_HQ_H5_PATH = os.path.join(BOSPHORUS_TEST_HQ_BASE_PATH, "bosphorus_test_HQ.h5")

# 2b) Results paths
# ______________________________________________________________________
RESULTS_LIGHT_PATH = os.path.join(PROJECT_ROOT, "results_light")
RESULTS_HEAVY_PATH = os.path.join(PROJECT_ROOT, "results_heavy")
# ______________________________________________________________________
ACCURACY_RESULTS_PATH = os.path.join(RESULTS_LIGHT_PATH, "accuracy_results")
CONSOLE_OUTPUTS_PATH = os.path.join(RESULTS_LIGHT_PATH, "console_outputs")

# 3) Model paths
# ______________________________________________________________________
MODELS_PATH = os.path.join(DATA_BASE_DIR, "models")
# ______________________________________________________________________
# > FEDERICA MODELS
FEDERICA_MODELS_FOLDER = os.path.join(MODELS_PATH, "federica")
FINETUNING_MODELS_FOLDER = os.path.join(FEDERICA_MODELS_FOLDER, "finetuning")
ALL_MODELS_PATHS = {
    "resnet_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ResNet_finetuning"),
    "pattlite_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_PattLite_finetuning"),
    "vgg19_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_VGG19_finetuning"),
    "inceptionv3_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_InceptionV3_finetuning"),
    "convnext_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ConvNeXt_finetuning"),
    "efficientnet_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_EfficientNetB1_finetuning_weights.h5"),
    "yolo_last": os.path.join(FEDERICA_MODELS_FOLDER, 'last.pt'),
}


# 4) Landmarks
# ______________________________________________________________________
# > MEDIAPIPE
MEDIAPIPE_PATH = os.path.join(MODELS_PATH, "mediapipe")
LANDMARKER_MODEL_PATH = os.path.join(MEDIAPIPE_PATH, "face_landmarker.task")
LANDMARK_COORDINATES_FOLDER = os.path.join(AUXILIARY_DATA_DIR, "landmark_coordinates")


if __name__ == "__main__":
    for model_name, model_path in ALL_MODELS_PATHS.items():
        # expand the path to abs path and make it clickable in console
        ALL_MODELS_PATHS[model_name] = os.path.abspath(model_path)
        print(f'{model_name}: "{ALL_MODELS_PATHS[model_name]}"')