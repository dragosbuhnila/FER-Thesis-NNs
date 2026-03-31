import os

# 0) Debug Options
SMALL_SUBSET_SIZE = 60000 												# if small_subset flag is True, will load only this many samples from the dataset. This is useful for debugging and for testing the data loading and training pipeline without having to wait for the entire dataset to load and train. Set to None to load the entire dataset.

DEBUG_MASKING = False                                                   # If True, will plot images during masking for debugging
DEBUG_MASKING_TIER2 = False if DEBUG_MASKING == False else True         # If True, will plot more images during masking for debugging

FORCE_RECALCULATE_LANDMARKS = False                                     # If True, will recalculate and overwrite existing landmark coordinates even if they exist

GLOBALS = {
    "TOTAL_IMAGES_LANDMARKS_SAVED": 0,                      			# Counter for total images saved during masking
    "TOTAL_IMAGES_LANDMARKS_LOADED": 0,                     			# Counter for total images loaded during masking
    "UNLANDMARKABLE_IMAGES_LIST": [],
	"DATALOADER_SHOW_IMAGES_B4AUG": False,
	"DATALOADER_SHOW_IMAGES_B4AUG_ONLYONCE": True,
	"DATALOADER_SHOW_IMAGES_FINAL": False,
	"DATALOADER_SHOW_IMAGES_FINAL_ONLYONCE": True,
	"DATALOADER_SHOW_IMAGES_SAVE_INSTEAD_OF_PLOT": False, 				# if True, will save the images instead of plotting them. This is useful for debugging in non-interactive environments and for keeping a record of the images that were shown during debugging
	"MAX_TIME_SPENT_IN_WHILE_LOOP_UNIFORM_MISMATCH": 0,
	"START_TIME": None,
	"SHORT_TRAINING_FOR_TESTING": False, 								# If True, epoch will last 1000 samples
	"FULL_EPOCHS": False, 												# If True, epoch will last circa like len(occluded_images) instead of len(original_images). The former is 250k+ images, the latter is 20k+ images. This is useful for debugging and for testing the data loading and training pipeline without having to wait for the entire dataset to load and train. Set to False to use the original images length for the epoch, which is what we want for actual training.
	"LONG_EPOCHS": False, 												# If True, epoch will last len(occluded_images) * (nof_types_of_occlusion // 2). This is a middle ground between full epochs and normal epochs.
}

# 1) Core Configurations
# ______________________________________________________________________
PROJECT_ROOT = 			os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# ______________________________________________________________________
DATA_BASE_DIR = 		os.path.join(PROJECT_ROOT, "data")
AUXILIARY_DATA_DIR = 	os.path.join(DATA_BASE_DIR, "auxiliary")
MODULES_DIR = 			os.path.join(PROJECT_ROOT, "modules")
MODELS_DIR = 			os.path.join(DATA_BASE_DIR, "models")
MLFLOW_DIR = 			os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_DB_WINDOWS = 	os.path.join(PROJECT_ROOT, "mlflow_windows.db")
OUT_ERR_DIR = 			os.path.join(PROJECT_ROOT, "out_err")
XAI_DIR = 				os.path.join(DATA_BASE_DIR, "xais_of_interest")

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
ADELE_TEST_SET_BASE_PATH = 				os.path.join(DATASETS_PATH, "adele_test_set")
ADELE_TEST_SET_H5_PATH = 				os.path.join(ADELE_TEST_SET_BASE_PATH, "adele_test_set.h5")
ADELE_TEST_SET_IMAGES_PATH = 			os.path.join(ADELE_TEST_SET_BASE_PATH, "extracted_images")
ADELE_TEST_SET_YAML_PATH = 				os.path.join(ADELE_TEST_SET_BASE_PATH, "adele_test_set_yaml")
# ____________________________________________________________________
# > ADELE_180ROTATED_TEST_SET
ADELE_180ROTATED_TEST_SET_BASE_PATH = 	os.path.join(DATASETS_PATH, "adele_180rotated_test_set")
ADELE_180ROTATED_TEST_SET_H5_PATH = 	os.path.join(ADELE_180ROTATED_TEST_SET_BASE_PATH, "adele_180rotated_test_set.h5")
ADELE_180ROTATED_TEST_SET_IMAGES_PATH = os.path.join(ADELE_180ROTATED_TEST_SET_BASE_PATH, "extracted_images")
# ______________________________________________________________________
# > OCCLUDED_TEST_SET
OCCLUDED_TEST_SET_BASE_PATH = 		os.path.join(DATASETS_PATH, "occluded_test_set")
OCCLUDED_TEST_SET_H5_PATH = 		os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "occluded_test_set.h5")
OCCLUDED_TEST_SET_IMAGES_PATH = 	os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "bosphorus_test_HQ")
OCCLUDED_TEST_SET_UNOCCLUDED_BACK_LARGE_IMAGES_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "bosphorus_test_HQ_unoccluded_back")
OCCLUDED_TEST_SET_RESIZED_PATH = 	os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "output_images_testset_resized")
OCCLUDED_TEST_SET_UNOCCLUDED_BACK_RESIZED_IMAGES_PATH = os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "unoccluded_back_images")
OCCLUDED_TEST_SET_YAML_PATH = 		os.path.join(OCCLUDED_TEST_SET_BASE_PATH, "occluded_test_set_yaml")
# ______________________________________________________________________
# > ORIGINAL_TRAIN_VAL_SET
ORIGINAL_TRAIN_VAL_SET_BASE_PATH = 			os.path.join(DATASETS_PATH, "original_train_val_set")
ORIGINAL_TRAIN_VAL_SET_H5_PATH = 			os.path.join(ORIGINAL_TRAIN_VAL_SET_BASE_PATH, "dataset.h5")
ORIGINAL_TRAIN_VAL_SET_IMAGES_BASE_PATH = 	os.path.join(ORIGINAL_TRAIN_VAL_SET_BASE_PATH, "dataset_extracted")
ORIGINAL_TRAIN_SET_IMAGES_PATH = 			os.path.join(ORIGINAL_TRAIN_VAL_SET_IMAGES_BASE_PATH, "train")
ORIGINAL_VAL_SET_IMAGES_PATH = 				os.path.join(ORIGINAL_TRAIN_VAL_SET_IMAGES_BASE_PATH, "val")
# ______________________________________________________________________
# > OCCLUDED_TRAIN_VAL_SET
OCCLUDED_TRAIN_VAL_SET_BASE_PATH = 	os.path.join(DATASETS_PATH, "occluded_train_val_set")
OCCLUDED_TRAIN_VAL_SET_H5_PATH = 	os.path.join(OCCLUDED_TRAIN_VAL_SET_BASE_PATH, "occluded_train_val_set.h5")
OCCLUDED_TRAIN_SET_IMAGES_PATH = 	os.path.join(OCCLUDED_TRAIN_VAL_SET_BASE_PATH, "images", "train")
OCCLUDED_VAL_SET_IMAGES_PATH = 		os.path.join(OCCLUDED_TRAIN_VAL_SET_BASE_PATH, "images", "val")
# ______________________________________________________________________
# > OCCLUDED + ORIGINAL TRAIN VAL SET (for YOLO)
OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_BASE_PATH = os.path.join(DATASETS_PATH, "occluded_and_original_trainval_occ8_set")
OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_YOLO_PATH = os.path.join(OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_BASE_PATH, "images")
OCCLUDED_AND_ORIGINAL_TRAIN_OCC8_SET_IMAGES_PATH = os.path.join(OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_BASE_PATH, "images", "train")
OCCLUDED_AND_ORIGINAL_VAL_OCC8_SET_IMAGES_PATH = os.path.join(OCCLUDED_AND_ORIGINAL_TRAIN_VAL_OCC8_SET_BASE_PATH, "images", "val")
# ______________________________________________________________________
# > BOSPHORUS_TEST_HQ                   # this h5 will be full size not resized to small images
BOSPHORUS_TEST_HQ_BASE_PATH = 		os.path.join(DATASETS_PATH, "bosphorus_test_HQ")
BOSPHORUS_TEST_HQ_IMAGES_PATH = 	os.path.join(BOSPHORUS_TEST_HQ_BASE_PATH, "bosphorus_test_HQ")
BOSPHORUS_TEST_HQ_H5_PATH = 		os.path.join(BOSPHORUS_TEST_HQ_BASE_PATH, "bosphorus_test_HQ.h5")

# 2b) Results paths
# ______________________________________________________________________
RESULTS_LIGHT_PATH = 				os.path.join(PROJECT_ROOT, "results_light")
RESULTS_HEAVY_PATH = 				os.path.join(PROJECT_ROOT, "results_heavy")
# ______________________________________________________________________
ACCURACY_RESULTS_PATH = 			os.path.join(RESULTS_LIGHT_PATH, "accuracy_results")
CONSOLE_OUTPUTS_PATH = 				os.path.join(RESULTS_LIGHT_PATH, "console_outputs")
SAVED_IMAGES_PATH = 				os.path.join(RESULTS_LIGHT_PATH, "saved_images")
# 3) Model paths
# ______________________________________________________________________
MODELS_PATH = 						os.path.join(DATA_BASE_DIR, "models")
# ______________________________________________________________________
# > FEDERICA AND OCCFT MODELS
FEDERICA_MODELS_FOLDER = 			os.path.join(MODELS_PATH, "federica")
FINETUNING_MODELS_FOLDER = 			os.path.join(FEDERICA_MODELS_FOLDER, "finetuning")

OCCFT_MODELS_FOLDER = 				os.path.join(MODELS_PATH, "occft")

# pattlite is actually mobilenet, but it was orignially called pattlite in federica's code (wrongly). When meeting either of the names, the architecture is always the same: mobilenetv1 bacbkone + pattlite upgrade
ALL_MODELS_PATHS = {
	# ----------------------------------- FEDERICA'S FINAL MODELS
    "resnet_finetuning": 			os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_ResNet_finetuning"),
    "pattlite_finetuning": 			os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_PattLite_finetuning"),
    "vgg19_finetuning": 			os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_VGG19_finetuning"),
    "inceptionv3_finetuning": 		os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_InceptionV3_finetuning"),
    "convnext_finetuning": 			os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_ConvNeXt_finetuning"),
    "efficientnet_finetuning": 		os.path.join(FINETUNING_MODELS_FOLDER, 	"pretrained_EfficientNetB1_finetuning_weights.h5"),
    "yolo_last": 					os.path.join(FEDERICA_MODELS_FOLDER,   	"last.pt"),
	# ----------------------------------- MY MODELS (occft -- i.e. occlusion finetuning)
	"occft_convnext": 				os.path.join(OCCFT_MODELS_FOLDER, "ConvNeXt_occfinetuning__20260216-225503__0.6771", 		"ConvNeXt_occfinetuning"),
    "occft_efficientnetb1":        	os.path.join(OCCFT_MODELS_FOLDER, "EfficientNetB1_occfinetuning__20260217-001742__0.6343", 	"EfficientNetB1_occfinetuning"),
    "occft_inceptionv3":           	os.path.join(OCCFT_MODELS_FOLDER, "InceptionV3_occfinetuning__20260212-212538__0.6743", 	"InceptionV3_occfinetuning"),
    "occft_pattlite":              	os.path.join(OCCFT_MODELS_FOLDER, "PattLite_occfinetuning__20260217-025925__0.6800", 		"PattLite_occfinetuning"),
    "occft_resnet":                	os.path.join(OCCFT_MODELS_FOLDER, "ResNet_occfinetuning__20260217-070852__0.6629", 			"ResNet_occfinetuning"),
    "occft_vgg19":                 	os.path.join(OCCFT_MODELS_FOLDER, "VGG19_occfinetuning__20260213-003832__0.7000", 			"VGG19_occfinetuning"),
	"occft_yolo":                 	os.path.join(OCCFT_MODELS_FOLDER, "yolo_last_occfinetuning__20260302-163800_cmplt-run_freeze-most_unfrozen__0.6886", "yolo_last_occfinetuning.pt"),
	# ----------------------------------- MY MODELS FULLRATIO 
	"resnet_special_fullratio":			os.path.join(OCCFT_MODELS_FOLDER, "ResNet_occfinetuning__20260308-165646__0.6343", "ResNet_occfinetuning"),
	"mobilenet_special_fullratio": 		os.path.join(OCCFT_MODELS_FOLDER, "PattLite_occfinetuning__20260308-164205__0.6371", "PattLite_occfinetuning"),
	"vgg_special_fullratio":			os.path.join(OCCFT_MODELS_FOLDER, "VGG19_occfinetuning__20260308-172924__0.7057", "VGG19_occfinetuning"),
	"inceptionv3_special_fullratio": 	os.path.join(OCCFT_MODELS_FOLDER, "InceptionV3_occfinetuning__20260308-173516__0.6429", "InceptionV3_occfinetuning"),
	"convnext_special_fullratio": 		os.path.join(OCCFT_MODELS_FOLDER, "ConvNeXt_occfinetuning__20260308-164148__0.6686", "ConvNeXt_occfinetuning"),
	"efficientnetb1_special_fullratio": os.path.join(OCCFT_MODELS_FOLDER, "EfficientNetB1_occfinetuning__20260308-174526__0.5943", "EfficientNetB1_occfinetuning"),
	# ----------------------------------- MY MODELS POSITIVE OCCLUSIONS ONLY
	"resnet_special_posonly":			os.path.join(OCCFT_MODELS_FOLDER, "ResNet_occfinetuning__20260309-133648__0.5286", "ResNet_occfinetuning"),
	"mobilenet_special_posonly": 		os.path.join(OCCFT_MODELS_FOLDER, "PattLite_occfinetuning__20260309-131714__0.4514", "PattLite_occfinetuning"),
	"vgg_special_posonly":				os.path.join(OCCFT_MODELS_FOLDER, "VGG19_occfinetuning__20260309-135404__0.5514", "VGG19_occfinetuning"),
	"inceptionv3_special_posonly": 		os.path.join(OCCFT_MODELS_FOLDER, "InceptionV3_occfinetuning__20260309-134158__0.5371", "InceptionV3_occfinetuning"),
	"convnext_special_posonly": 		os.path.join(OCCFT_MODELS_FOLDER, "ConvNeXt_occfinetuning__20260309-131814__0.5486", "ConvNeXt_occfinetuning"),
	"efficientnetb1_special_posonly": 	os.path.join(OCCFT_MODELS_FOLDER, "EfficientNetB1_occfinetuning__20260309-133425__0.4971", "EfficientNetB1_occfinetuning"),
}

OCCFT_MODELS_PATHS = {model_name: model_path for model_name, model_path in ALL_MODELS_PATHS.items() if "occft" in model_name.lower()}

# 3b) Evaluations
# ______________________________________________________________________
EVALUATIONS_FOLDER_PATH = os.path.join(DATA_BASE_DIR, "evaluations")
OCCFT_MODELS_RESULTS_FOLDER = os.path.join(EVALUATIONS_FOLDER_PATH, "occft_models")

OCCFT_MODELS_RESULTS_PATHS = {}
for model_name, tf_model_path in ALL_MODELS_PATHS.items():
	if "occft" in model_name.lower():
		model_folder_path = os.path.abspath(os.path.join(tf_model_path, "..")) # get the folder containing the model
		model_folder_name = os.path.basename(model_folder_path)
		model_evaluation_results_path = os.path.join(OCCFT_MODELS_RESULTS_FOLDER, model_folder_name)
		OCCFT_MODELS_RESULTS_PATHS[model_name] = model_evaluation_results_path


# 4) Landmarks
# ______________________________________________________________________
# > MEDIAPIPE
MEDIAPIPE_PATH = 					os.path.join(MODELS_PATH, "mediapipe")
LANDMARKER_MODEL_PATH = 			os.path.join(MEDIAPIPE_PATH, "face_landmarker.task")
LANDMARK_COORDINATES_FOLDER_PATH = 	os.path.join(AUXILIARY_DATA_DIR, "landmark_coordinates")
LANDMARK_COORDINATES_CACHE_EXPECTED_SIZE = 26951 # expected number of landmark coordinate files. If the size of the db changes this should change accordingly
CANONICAL_FACES_CUT = 				os.path.join(AUXILIARY_DATA_DIR, "canonical_faces_cut")


# 5) Datasets
# ______________________________________________________________________
# > images shape
IMAGES_SHAPE = (128, 128, 3)
# ______________________________________________________________________
# > Bosphorus Dupes
#       these hashes aren't actual conflicts, the duplication warning is triggered due to there being the same image twice in the training set
#           so I'll provide also the indices of the second appearance so they can be thrown out at will in the dataloader.
#       Use scripts/tools/show_specific_dataset_image_by_index.py with "show_trainset_dupes.py"
BOSPHORUS_DUPLICATE_IMAGES = {          # split,    index1, index2
    "4f50f6cba30511ed1a5731121979986c": {"split": "X_train", "ok_idx": 526,  "idx_to_remove": 527},        
    "1cd575ac2bb8e82ef46ad3758d64b308": {"split": "X_train", "ok_idx": 5526, "idx_to_remove": 5527},      
    "015958af0120539c5a911df3ad77f6f8": {"split": "X_train", "ok_idx": 5136, "idx_to_remove": 21269},      
    "6b3e14e0646a97eb1ff84f66f4896d96": {"split": "X_train", "ok_idx": 5137, "idx_to_remove": 21270},      
}

BOSPHORUS_UNLANDMARKABLE_IMAGES = {
	"20eb13a5aee7da2826e3f3db18a2ba70": {"split": "X_train", "idx_to_remove": 19},
	"86c35fb75d7e4fcb9e09ba9c620b2425": {"split": "X_train", "idx_to_remove": 457},
	"27429ebc6ede601f8471f160e3040ed8": {"split": "X_train", "idx_to_remove": 741},
	"dc30fed59f51748ae7bc1be977c55107": {"split": "X_train", "idx_to_remove": 862},
	"6aafd22b84924d5c9d2aa6394dd89f84": {"split": "X_train", "idx_to_remove": 980},
	"c2f17e4aa269fdb28eb314967b83c93b": {"split": "X_train", "idx_to_remove": 1214},
	"d3d2edf3abb8675c2ee8e349a8ce9d59": {"split": "X_train", "idx_to_remove": 1318},
	"a9cfce94d71fff52741583a36f9cb92a": {"split": "X_train", "idx_to_remove": 1416},
	"c1e5a70e9f55f304d527af14f4fa679a": {"split": "X_train", "idx_to_remove": 1651},
	"e6ab1d2a3c17a31ef82e07c9e46dd5c0": {"split": "X_train", "idx_to_remove": 1699},
	"7db37b671f4ea6fc06d5a33398c4042a": {"split": "X_train", "idx_to_remove": 2332},
	"86d7a6c82cc92155a1a92f5c8d2ea92a": {"split": "X_train", "idx_to_remove": 3359},
	"717d41feee3dbd825712129a00adfe16": {"split": "X_train", "idx_to_remove": 4651},
	"d03e2d410ac6592b018f7a10497d3b0f": {"split": "X_train", "idx_to_remove": 4934},
	"38561209f4703eafd91f07327fe74a09": {"split": "X_train", "idx_to_remove": 8802},
	"4259ff371a66691a5d2ec99f7e0a3b52": {"split": "X_train", "idx_to_remove": 8814},
	"8540078cfeac2a618c54df510248d819": {"split": "X_train", "idx_to_remove": 9076},
	"db67c87f98988aaa14844e2183e966fd": {"split": "X_train", "idx_to_remove": 9565},
	"9fe8900abba6cd64afad4b155789495e": {"split": "X_train", "idx_to_remove": 10472},
	"0df37033bc2f23695b8bc1e66476e527": {"split": "X_train", "idx_to_remove": 12093},
	"6c42312f38c41c29a80c4f28c4549790": {"split": "X_train", "idx_to_remove": 12418},
	"b3285a23d854a2153208aa1554b3aa43": {"split": "X_train", "idx_to_remove": 16971},
	"82fbfb28fb8f44fa8fd8b323c43057d3": {"split": "X_train", "idx_to_remove": 18578},
	"94c95827591da2e44ad66ab33994e3ac": {"split": "X_train", "idx_to_remove": 19293},
	"c4083dce4d10d36835dbd2f437ac0fee": {"split": "X_train", "idx_to_remove": 19997},
    
	"e9b07df10364d2912359aeee15dd62f4": {"split": "X_val", "idx_to_remove": 205},
	"c94bd6addf858f77043362c4bc7d0efc": {"split": "X_val", "idx_to_remove": 422},
	"44f72201d170ba615d0ba0330e264ec7": {"split": "X_val", "idx_to_remove": 497},
	"2fdaa0e6835a3a785f3038d169c3d97d": {"split": "X_val", "idx_to_remove": 512},
	"bcc9b8c09f2c1c8ee8b93732f3e1f0df": {"split": "X_val", "idx_to_remove": 903},
	"5878ab74eebb3523a8755767674096ae": {"split": "X_val", "idx_to_remove": 945},
	"46e9d2b3d2f8c6a1d9d6bdd838de39d6": {"split": "X_val", "idx_to_remove": 2909},
	"612a10a1e3d726a9a97d5ae83fd49024": {"split": "X_val", "idx_to_remove": 4735},
}

config_do_remove_ok_idx_too = True # if True, will remove the ok_idx from the dataset as well, to be safe. This is because duplicate images may or do have conflicting labels

BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT = {}
for images_dict in [BOSPHORUS_DUPLICATE_IMAGES, BOSPHORUS_UNLANDMARKABLE_IMAGES]:
	for image_hash, image_data in images_dict.items():
		split = image_data["split"]
		idx_to_remove = image_data["idx_to_remove"]
		ok_idx = image_data.get("ok_idx", None)
		if split not in BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT:
			BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT[split] = []
		BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT[split].append(idx_to_remove)
		if config_do_remove_ok_idx_too and ok_idx is not None:
			BOSPHORUS_INDICES_TO_REMOVE_BY_SPLIT[split].append(ok_idx)


# 6) XAI
HEATMAPS_DIR_PATH = os.path.join(XAI_DIR, "HEATMAPS")
CONFUSION_MATRICES_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "CONFUSION_MATRICES")
BUBBLES_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "Bubbles")
EXTERNAL_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "EXTERNAL")
GRADCAM_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "GRADCAM")
GRADCAM_LAYERS_PDFS_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "GRADCAM_LAYERS_PDFS")
MERGED_HEATMAPS_PDFS_DIR_PATH = os.path.join(HEATMAPS_DIR_PATH, "MERGED_HEATMAPS_PDFS")


# 7) other folders
DOCS_FOLDER = os.path.join(PROJECT_ROOT, "docs")
MODEL_DOCS_FOLDER = os.path.join(DOCS_FOLDER, "models")



if __name__ == "__main__":
    for model_name, model_path in ALL_MODELS_PATHS.items():
        # expand the path to abs path and make it clickable in console
        ALL_MODELS_PATHS[model_name] = os.path.abspath(model_path)
        print(f'{model_name}: "{ALL_MODELS_PATHS[model_name]}"')