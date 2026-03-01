import os; import sys;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import ALL_MODELS_PATHS
from modules.yolo import load_yolo_model



MODEL_NAME = "yolo_last"



if __name__ == "__main__":
    if MODEL_NAME not in ALL_MODELS_PATHS:
        raise ValueError(f"Model name '{MODEL_NAME}' not found in ALL_MODELS_PATHS. Please check the model name and ensure it is defined in the configuration.")
    model_path = ALL_MODELS_PATHS[MODEL_NAME]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure the model file exists at the specified path.")
    print(f"Model '{MODEL_NAME}' found at path: {model_path}. The YOLO architecture seems to be correctly set up.")

    model = load_yolo_model(MODEL_NAME)
    net = model.model

    # print number of layers
    total_modules = sum(1 for _ in net.modules())
    print("Total modules:", total_modules)
    total_param_layers = sum(1 for m in net.modules() if len(list(m.parameters())) > 0)
    print("Layers with parameters:", total_param_layers)

    # print all layers
    for i, (name, module) in enumerate(net.named_modules()):
        print(f"{i:3} | {name} | {module.__class__.__name__}")

    # # print only param layers
    # for i, (name, module) in enumerate(net.named_modules()):
    #     if any(p.requires_grad for p in module.parameters(recurse=False)):
    #         print(f"{i:3} | {name} | {module.__class__.__name__}")
