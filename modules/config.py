# 1) Emotions
import os
DATA_BASE_DIR = os.path.join(".", "data")

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

# 2) Paths

# 2a) Datasets paths
DATASETS_PATH = os.path.join(DATA_BASE_DIR, "datasets")

OCCLUDED_TEST_SET_BASE_PATH = os.path.join(DATASETS_PATH, "occluded_test_set")
OCCLUDED_TEST_SET_H5_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "occluded_test_set.h5")
OCCLUDED_TEST_SET_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "bosphorus_test_HQ")
OCCLUDED_TEST_SET_RESIZED_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "output_images_testset_resized")

ADELE_TEST_SET_BASE_PATH = os.path.join(DATASETS_PATH, "adele_test_set")
ADELE_TEST_SET_H5_PATH = os.path.join(ADELE_TEST_SET_BASE_PATH, "adele_test_set.h5")

# 2b) Results paths
RESULTS_LIGHT_PATH = os.path.join(".", "results_light")
RESULTS_HEAVY_PATH = os.path.join(".", "results_heavy")

ACCURACY_RESULTS_PATH = os.path.join(RESULTS_LIGHT_PATH, "accuracy_results")

# 3) Model paths
MODELS_PATH = os.path.join(DATA_BASE_DIR, "models")
FINETUNING_MODELS_FOLDER = os.path.join(MODELS_PATH, "federica", "finetuning")
ALL_MODELS_PATHS = {
    "resnet_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ResNet_finetuning"),
    "pattlite_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_PattLite_finetuning"),
    "vgg19_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_VGG19_finetuning"),
    "inceptionv3_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_InceptionV3_finetuning"),
    "convnext_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_ConvNeXt_finetuning"),
    "efficientnet_finetuning": os.path.join(FINETUNING_MODELS_FOLDER, "pretrained_EfficientNetB1_finetuning_weights.h5"),
}


if __name__ == "__main__":
    for model_name, model_path in ALL_MODELS_PATHS.items():
        # expand the path to abs path and make it clickable in console
        ALL_MODELS_PATHS[model_name] = os.path.abspath(model_path)
        print(f'{model_name}: "{ALL_MODELS_PATHS[model_name]}"')