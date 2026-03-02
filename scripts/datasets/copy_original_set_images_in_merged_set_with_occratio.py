import os; import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tqdm import tqdm
import shutil
import math

from modules.config import OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH, \
                            ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH, \
                            EMOTIONS

OCC_RATIO = 0.8
OCCLUDED_TRAIN_SET_IMAGE_COUNT = 236000

COPY_FROM_TO = {
    ORIGINAL_TRAIN_SET_IMAGES_PATH: OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH,
    ORIGINAL_VAL_SET_IMAGES_PATH: OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH
}



def calculate_images_amount(occluded_set_path):
    total_occluded_images = 0
    for emotion in EMOTIONS:
        emotion_path = os.path.join(occluded_set_path, emotion)
        num_images = len(os.listdir(emotion_path))
        total_occluded_images += num_images

    return total_occluded_images



if __name__ == '__main__':
    # 0) First check file structure
    for path in [OCCLUDED_AND_ORIGINAL_TRAIN_SET_IMAGES_PATH, OCCLUDED_AND_ORIGINAL_VAL_SET_IMAGES_PATH,
                 ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH]:
        if not os.path.exists(path):
            print(f'Error: Path {path} does not exist.')
            sys.exit(1)
        
        for emotion in EMOTIONS:
            emotion_path = os.path.join(path, emotion)
            if not os.path.exists(emotion_path):
                print(f'Error: Path {emotion_path} does not exist.')
                sys.exit(1)

    # 1) Calculate the duplication ratio needed for the specific occlusion ratio (keep it an integer)
    # occ_ratio of 0.8 means unocc ratio of 0.2. x is the number of desired unoccluded images, and the number of occluded images
    # x/(A+x)=unocc => 
    # x = A*unocc + unocc*x => 
    # occ*x = A*unocc => 
    # x = A*(unocc/occ) =>
    # x = A*((1-occ)/occ)
    unocc_to_occ_ratio = (1 - OCC_RATIO) / OCC_RATIO
    if OCC_RATIO == 0.8 and not math.isclose(unocc_to_occ_ratio, 0.25):
        print(f'Error: Expected unocc_to_occ_ratio of 0.25 for OCC_RATIO of 0.8, got {unocc_to_occ_ratio}')
        sys.exit(1)
    if OCC_RATIO == 0.5 and not math.isclose(unocc_to_occ_ratio, 1):
        print(f'Error: Expected unocc_to_occ_ratio of 1 for OCC_RATIO of 0.5, got {unocc_to_occ_ratio}')
        sys.exit(1)

    # Copy images from original set to merged set
    for original_set_path in [ORIGINAL_TRAIN_SET_IMAGES_PATH, ORIGINAL_VAL_SET_IMAGES_PATH]:
        occluded_set_path = COPY_FROM_TO[original_set_path]

        print(f"Counting total images in original and occluded relative to {occluded_set_path}")
        total_occluded_images = calculate_images_amount(occluded_set_path) # e.g. 236286 for train set, 58418 for val set
        total_original_images = calculate_images_amount(original_set_path) # e.g.  21332 for train set,  5273 for val set
        print(f'Total occluded images: {total_occluded_images}, total original images: {total_original_images}')

        target_unoccluded_images = int(total_occluded_images * unocc_to_occ_ratio) # e.g. 59071 for train set, 14605 for val set
        if target_unoccluded_images > total_original_images:
            duplication_factor = round(target_unoccluded_images / total_original_images) # e.g. 3 for train set, 3 for val set
        else:
            raise ValueError(f'Error: Target unoccluded images ({target_unoccluded_images}) is less than or equal to total original images ({total_original_images}), no duplication needed. You should instead remove some')

        for emotion in EMOTIONS:
            original_emotion_path = os.path.join(original_set_path, emotion)
            merged_emotion_path = os.path.join(occluded_set_path, emotion)

            # Get list of images in original emotion folder
            original_images = [f for f in os.listdir(original_emotion_path) if os.path.isfile(os.path.join(original_emotion_path, f))]

            # Copy images with duplication
            for i in range(duplication_factor):
                for image_name in tqdm(original_images, desc=f'Copying {emotion} images, round {i+1}/{duplication_factor}', unit='image'):
                    src_image_path = os.path.join(original_emotion_path, image_name)
                    if i == 0:
                        dst_image_name = f"{image_name}_unoccluded"
                    else:
                        dst_image_name = f'{os.path.splitext(image_name)[0]}_unoccluded_dup{i}{os.path.splitext(image_name)[1]}'
                    dst_image_path = os.path.join(merged_emotion_path, dst_image_name)
                    shutil.copy(src_image_path, dst_image_path)
                    # print(f'Copying {src_image_path} to {dst_image_path}')
