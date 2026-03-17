import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import CONSOLE_OUTPUTS_PATH, XAI_DIR
from modules.misc import Tee, get_timestamp



JUST_TEST_WITH_PRINTS = False


MODEL_NAMES = ["occft_convnext", "occft_efficientnetb1", "occft_inceptionv3", "occft_pattlite", "occft_resnet", "occft_vgg19", "occft_yolo"]
HEATMAPS_DIR_BASENAME = "HEATMAPS"
HEATMAPS_DIR_PATH = os.path.join(XAI_DIR, HEATMAPS_DIR_BASENAME)
BUBBLES_DIR_BASENAME = "Bubbles"
EXTERNAL_DIR_BASENAME = "EXTERNAL"
GRADCAM_DIR_BASENAME = "GRADCAM"


LOG_FILE_PATH = os.path.join(CONSOLE_OUTPUTS_PATH, f"{get_timestamp()}__move_heatmaps.log")
log_dir = os.path.dirname(LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
sys.stdout = Tee(LOG_FILE_PATH)
sys.stderr = Tee(LOG_FILE_PATH) 


if __name__ == "__main__":
    # for model_name in MODEL_NAMES:
    #     # make a dir witthe model name in the HEATMAPS_DIR_PATH
    #     model_dir = os.path.join(HEATMAPS_DIR_PATH, model_name)
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)

    for root, dirs, files in os.walk(XAI_DIR):
        if os.path.basename(root) == HEATMAPS_DIR_BASENAME:
            source_path = root
            if "bubbles" in root.lower():
                source_paths = [os.path.join(root, d) for d in dirs]
                dest_paths = [os.path.join(HEATMAPS_DIR_PATH, BUBBLES_DIR_BASENAME, f"{d.split('_')[2]}_{d.split('_')[3]}") for d in dirs]
                for src, dst in zip(source_paths, dest_paths):
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                    for file in os.listdir(src):
                        src_file = os.path.join(src, file)
                        dst_file = os.path.join(dst, file)
                        if not JUST_TEST_WITH_PRINTS:
                            os.rename(src_file, dst_file)
                        print(f"Source: {src_file}")
                        print(f"Destination: {dst_file}")
                        print()

            elif "extpert" in root.lower() or "external" in root.lower():
                dest_path = os.path.join(HEATMAPS_DIR_PATH, EXTERNAL_DIR_BASENAME)
                dest_path = os.path.join(dest_path, os.path.basename(os.path.dirname(root)))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                for file in files:
                    src_file = os.path.join(source_path, file)
                    dest_file = os.path.join(dest_path, file)
                    if not JUST_TEST_WITH_PRINTS:
                        os.rename(src_file, dest_file)
                    print(f"Source: {src_file}")
                    print(f"Destination: {dest_file}")
                    print()

            elif "gradcam" in root.lower():
                dest_path = os.path.join(HEATMAPS_DIR_PATH, GRADCAM_DIR_BASENAME)
                dest_path = os.path.join(dest_path, os.path.basename(os.path.dirname(os.path.dirname(root))), os.path.basename(os.path.dirname(root)))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                for file in files:
                    src_file = os.path.join(source_path, file)
                    dest_file = os.path.join(dest_path, file)
                    if not JUST_TEST_WITH_PRINTS:
                        os.rename(src_file, dest_file)
                    print(f"Source: {src_file}")
                    print(f"Destination: {dest_file}")
                    print()

