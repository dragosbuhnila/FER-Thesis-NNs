import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import h5py
import json

from modules.config import ALL_MODELS_PATHS



RUN_LAYERS = True
RUN_COMPLETE = False
if not RUN_LAYERS and not RUN_COMPLETE:
    print("[WARNING] No operation selected. Please set RUN_LAYERS or RUN_COMPLETE to True.")
    sys.exit(0)



path = ALL_MODELS_PATHS["convnext_finetuning"]
print(f"[INFO] Reading HDF5 file at: {path}")

if RUN_LAYERS:
    with h5py.File(path, "r") as f:
        model_config = json.loads(f.attrs["model_config"])
        layers = model_config["config"]["layers"]

        for l in layers:
            if "layer_scale" in l["config"]["name"]:
                print(json.dumps(l, indent=2))
            print(f"[INFO] Layer name: {l['config']['name']} - Type: {l['class_name']}")
        print(f"[INFO] Total layers: {len(layers)}")

if RUN_COMPLETE:
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name, obj.shape)

        f.visititems(walk)

