from copy import deepcopy

from Exp.ERA5.runtime_config import DEFAULT_MODEL_CFG
from Model.ConvLSTM import ConvLSTMModel
from Model.SwinTrans import SwinTransModel
from Model.UniPhy.ModelUniPhy import UniPhyModel

from .components import (
    NoTimeDtWrapper,
    apply_fixed_scale_weights,
    apply_single_scale_mixer,
)
from .protocol import VARIANT_ORDER, get_variant_spec


def _normalize_cfg(model_cfg: dict) -> dict:
    cfg = deepcopy(DEFAULT_MODEL_CFG)
    cfg.update({k: v for k, v in model_cfg.items() if v is not None})
    cfg["patch_size"] = tuple(cfg["patch_size"])
    return cfg


def _build_base_model(model_cfg: dict, device=None) -> UniPhyModel:
    cfg = _normalize_cfg(model_cfg)
    model = UniPhyModel(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        embed_dim=int(cfg["embed_dim"]),
        expand=int(cfg["expand"]),
        depth=int(cfg["depth"]),
        patch_size=tuple(cfg["patch_size"]),
        img_height=int(cfg["img_height"]),
        img_width=int(cfg["img_width"]),
        dt_ref=float(cfg["dt_ref"]),
    )
    if device is not None:
        model = model.to(device)
    return model


def _build_swin_model(model_cfg: dict, device=None) -> SwinTransModel:
    cfg = _normalize_cfg(model_cfg)
    model = SwinTransModel(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        embed_dim=int(cfg["embed_dim"]),
        depth=int(cfg["depth"]),
        patch_size=tuple(cfg["patch_size"]),
        img_height=int(cfg["img_height"]),
        img_width=int(cfg["img_width"]),
    )
    if device is not None:
        model = model.to(device)
    return model


def _build_convlstm_model(model_cfg: dict, device=None) -> ConvLSTMModel:
    cfg = _normalize_cfg(model_cfg)
    hidden_dim = max(16, int(round(int(cfg["embed_dim"]) * 0.55)))
    model = ConvLSTMModel(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        embed_dim=hidden_dim,
        depth=int(cfg["depth"]),
        patch_size=tuple(cfg["patch_size"]),
        img_height=int(cfg["img_height"]),
        img_width=int(cfg["img_width"]),
    )
    if device is not None:
        model = model.to(device)
    return model


TRAINING_ONLY_VARIANTS = {"E1_l1_only"}


VARIANTS = {
    "baseline": (
        "Active deterministic UniPhy model.",
        lambda cfg, dev: _build_base_model(cfg, dev),
    ),
    "A1_no_dt": (
        "Every time interval is replaced by dt_ref.",
        lambda cfg, dev: NoTimeDtWrapper(_build_base_model(cfg, dev)),
    ),
    "D1_single_scale": (
        "The spatial mixer keeps only the local branch.",
        lambda cfg, dev: apply_single_scale_mixer(_build_base_model(cfg, dev)),
    ),
    "D2_fixed_scale_weights": (
        "The adaptive spatial scale gate is fixed to uniform logits.",
        lambda cfg, dev: apply_fixed_scale_weights(_build_base_model(cfg, dev)),
    ),
    "E1_l1_only": (
        "The deterministic objective uses L1 only.",
        lambda cfg, dev: _build_base_model(cfg, dev),
    ),
    "G1_swin_transformer": (
        "Swin style fixed interval single frame predictor.",
        lambda cfg, dev: _build_swin_model(cfg, dev),
    ),
    "G2_convlstm": (
        "ConvLSTM fixed interval recurrent predictor.",
        lambda cfg, dev: _build_convlstm_model(cfg, dev),
    ),
}


def build_variant(variant: str, model_cfg: dict, device=None):
    variant = str(variant).strip()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {sorted(VARIANTS)}")
    return VARIANTS[variant][1](model_cfg, device)


def list_variants(include_baseline=True):
    names = list(VARIANT_ORDER)
    if include_baseline:
        return names
    return [name for name in names if name != "baseline"]


def describe_variant(variant: str) -> dict:
    spec = get_variant_spec(variant).to_dict()
    spec["implementation"] = VARIANTS[variant][0]
    return spec


def build_variant_optimizer(model, cfg: dict, variant: str):
    del variant
    from Exp.ERA5.runtime_config import build_adamw_optimizer

    return build_adamw_optimizer(model, cfg)
