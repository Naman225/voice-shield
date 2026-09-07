import os
import torch
import torch.nn as nn

# Import the vendored AASIST architecture directly from this package.
# aasist_core.py is a verbatim copy of the original AASIST.py from NAVER Corp.
# (MIT License, Copyright (c) 2021-present NAVER Corp.)
from .aasist_core import Model  # noqa: E402

# Default to the fine-tuned checkpoint stored inside the repo's checkpoints/ folder.
# Falls back to the original upstream AASIST weights if the fine-tuned file is missing.
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_MODELS_DIR, "..", ".."))

DEFAULT_CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "best_indiefake_aasist.pth")
FALLBACK_CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "prototype_aasist_v1.pth")

# Standard AASIST model configuration (matches the published paper / upstream repo)
AASIST_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def load_pretrained_aasist(checkpoint_path=None, device="cpu"):
    """Load the AASIST model from a checkpoint file.

    Args:
        checkpoint_path: Path to a .pth state-dict file. Defaults to
            ``checkpoints/best_indiefake_aasist.pth`` inside the repo root.
        device: PyTorch device string (e.g. "cuda" or "cpu").

    Returns:
        torch.nn.Module: The AASIST model loaded with the given weights, moved
        to ``device`` and set to eval mode.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = FALLBACK_CHECKPOINT

    model = Model(AASIST_CONFIG)

    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[AASIST] ✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[AASIST] ⚠️  Checkpoint not found at: {checkpoint_path} — running with random weights.")

    return model.to(device).eval()


def get_optimizer_groups(model, frontend_lr=1e-6, backend_lr=3e-5):
    """Split model parameters into frontend (raw waveform encoder) and backend groups.

    This enables differential learning rates: a smaller LR for the pre-trained
    sinc-conv frontend and a larger LR for the graph attention backend.

    Args:
        model: An AASIST Model instance.
        frontend_lr: Learning rate for the frontend (conv_time, first_bn, encoder).
        backend_lr: Learning rate for the remaining (backend) parameters.

    Returns:
        list[dict]: A list of parameter group dicts compatible with any
        ``torch.optim.Optimizer``.
    """
    frontend_params = (
        list(model.conv_time.parameters())
        + list(model.first_bn.parameters())
        + list(model.encoder.parameters())
    )
    frontend_ids = {id(p) for p in frontend_params}
    backend_params = [p for p in model.parameters() if id(p) not in frontend_ids]

    return [
        {"params": frontend_params, "lr": frontend_lr, "weight_decay": 1e-4},
        {"params": backend_params, "lr": backend_lr, "weight_decay": 1e-4},
    ]