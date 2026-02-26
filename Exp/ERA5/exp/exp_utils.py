import os
import sys

import torch

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel

VALID_ARGS = {
    "in_channels",
    "out_channels",
    "embed_dim",
    "expand",
    "depth",
    "patch_size",
    "img_height",
    "img_width",
    "dt_ref",
    "sde_mode",
    "init_noise_scale",
    "ensemble_size",
}

DEFAULT_CFG = {
    "in_channels": 30,
    "out_channels": 30,
    "embed_dim": 256,
    "expand": 4,
    "depth": 8,
    "patch_size": [7, 15],
    "img_height": 721,
    "img_width": 1440,
    "dt_ref": 6.0,
    "sde_mode": "sde",
    "init_noise_scale": 0.0001,
    "ensemble_size": 4,
}


def load_config_and_model(ckpt_path, device, allow_missing=False):
    if not os.path.exists(ckpt_path):
        if allow_missing:
            return None, None
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cfg" in checkpoint and "model" in checkpoint["cfg"]:
        model_cfg = dict(checkpoint["cfg"]["model"])
    else:
        model_cfg = dict(DEFAULT_CFG)
    state_dict = checkpoint.get("model", checkpoint)
    if "encoder.pos_emb" in state_dict:
        model_cfg["embed_dim"] = state_dict["encoder.pos_emb"].shape[1]
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in VALID_ARGS}
    if "patch_size" in filtered_cfg:
        filtered_cfg["patch_size"] = tuple(filtered_cfg["patch_size"])
    model = UniPhyModel(**filtered_cfg).to(device)
    clean_state = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model, model_cfg


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
