import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import PROJECT_ROOT



# saliencies_folder = r"C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\d3 Masks\saliency_maps\canonical\gradcam\gradcam"
gradcam_saliencies_folder = os.path.join(PROJECT_ROOT, "..", "d3 Masks", "saliency_maps", "canonical", "gradcam", "gradcam")



if __name__ == "__main__":
    model_dirs = [d for d in os.listdir(gradcam_saliencies_folder) if os.path.isdir(os.path.join(gradcam_saliencies_folder, d))]
    model_name_to_layer_names = {}
    for model_dir in model_dirs:
        model_name_to_layer_names[model_dir] = []

        model_path = os.path.join(gradcam_saliencies_folder, model_dir)
        print(f"Processing model directory: {model_path}")
        
        layers_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        print(f"Found {len(layers_dirs)} layer directories")
        for layer_dir in layers_dirs:
            if "corretto" not in layer_dir:
                raise ValueError(f"Unexpected layer directory name: {layer_dir}. Expected to contain 'corretto'.")
        
            # from something like base_name_layer_10_corretto or base_name_layer_9_corretto, remove _layer_10_corretto or _layer_9_corretto 
            layer_name = layer_dir.split("_layer")[0]
            print(f" \t{layer_name}")
            
