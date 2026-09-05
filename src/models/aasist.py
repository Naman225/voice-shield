import os
import sys
import importlib.util
import torch
import torch.nn as nn

# Locate project root and aasist directory automatically
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir: /home/naman/Desktop/SIH/IndieFake/src/models
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
aasist_dir = os.path.join(project_root, "aasist")

# Directly import AASIST.py by file location to prevent package name collisions on 'models'
aasist_file = os.path.join(aasist_dir, "models", "AASIST.py")
if not os.path.isfile(aasist_file):
    raise FileNotFoundError(f"Cannot find AASIST model file at: {aasist_file}")

spec = importlib.util.spec_from_file_location("aasist_model_core", aasist_file)
aasist_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aasist_core)
Model = aasist_core.Model

DEFAULT_CHECKPOINT = os.path.join(aasist_dir, "models", "weights", "AASIST.pth")

def load_pretrained_aasist(checkpoint_path=None, device="cuda"):
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    model = Model(config)
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[AASIST] Successfully loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[AASIST] Warning: Checkpoint not found at {checkpoint_path}")

    return model.to(device)

def get_optimizer_groups(model, frontend_lr=1e-6, backend_lr=3e-5):
    frontend_params = list(model.conv_time.parameters()) + \
                      list(model.first_bn.parameters()) + \
                      list(model.encoder.parameters())
    frontend_ids = set(id(p) for p in frontend_params)
   
    backend_params = [p for p in model.parameters() if id(p) not in frontend_ids]
    return [
        {"params": frontend_params, "lr": frontend_lr, "weight_decay": 1e-4},
        {"params": backend_params,  "lr": backend_lr,  "weight_decay": 1e-4}
    ]
