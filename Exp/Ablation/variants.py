from copy import deepcopy

from Exp.ERA5.runtime_config import DEFAULT_MODEL_CFG
from Model.UniPhy.ModelUniPhy import UniPhyModel

from .components import (
    NoTimeDtWrapper,
    DeterministicWrapper,
    apply_etd1_integrator,
    apply_fixed_decay,
    apply_fixed_scale_weights,
    apply_constant_readout,
    apply_readout_residual,
    apply_single_scale_mixer,
)
from .protocol import VARIANT_ORDER, VARIANT_SPECS, get_variant_spec


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
        init_noise_scale=float(cfg["init_noise_scale"]),
        latent_dynamics=str(cfg.get("latent_dynamics", "real")),
    )
    if device is not None:
        model = model.to(device)
    return model


def _build_complex_model(model_cfg: dict, device=None) -> UniPhyModel:
    cfg = _normalize_cfg(model_cfg)
    cfg["latent_dynamics"] = "complex"
    return _build_base_model(cfg, device=device)


TRAINING_ONLY_VARIANTS = {"E1_l1_only"}

VARIANTS: dict = {
    "baseline": (
        "Full UniPhy model, no modifications.",
        lambda cfg, dev: _build_base_model(cfg, dev),
    ),
    "A1_no_dt": (
        "Variable dt replaced by constant dt_ref everywhere.",
        lambda cfg, dev: NoTimeDtWrapper(_build_base_model(cfg, dev)),
    ),
    "A2_discrete_rnn": (
        "SDE propagator replaced by a plain Elman RNN (dt-agnostic).",
        lambda cfg, dev: _build_discrete_rnn(cfg, dev),
    ),
    "B1_complex_latent": (
        "Dissipative real latent replaced by the previous complex eigenspace pathway.",
        lambda cfg, dev: _build_complex_model(cfg, dev),
    ),
    "B2_fixed_decay": (
        "Learned continuous decay spectrum replaced by a fixed reference decay.",
        lambda cfg, dev: apply_fixed_decay(_build_base_model(cfg, dev)),
    ),
    "C1_deterministic": (
        "Stochastic term zeroed; fully deterministic dynamics.",
        lambda cfg, dev: DeterministicWrapper(_build_base_model(cfg, dev)),
    ),
    "C2_no_readout_residual": (
        "Time-decayed latent readout residual added to each temporal block.",
        lambda cfg, dev: apply_readout_residual(_build_base_model(cfg, dev)),
    ),
    "C3_constant_readout": (
        "Readout residual kept constant across physical lead time.",
        lambda cfg, dev: apply_constant_readout(_build_base_model(cfg, dev)),
    ),
    "D1_single_scale": (
        "MultiScaleSpatialMixer uses only the local 3×3 branch.",
        lambda cfg, dev: apply_single_scale_mixer(_build_base_model(cfg, dev)),
    ),
    "D2_fixed_scale_weights": (
        "Multi-scale spatial weights fixed to uniform branches.",
        lambda cfg, dev: apply_fixed_scale_weights(_build_base_model(cfg, dev)),
    ),
    "E1_l1_only": (
        "L1 loss only; CRPS term removed from training objective.",
        lambda cfg, dev: _build_base_model(cfg, dev),
    ),
    "F1_etd1_integrator": (
        "Exact dissipative transition replaced by a first-order Euler transition.",
        lambda cfg, dev: apply_etd1_integrator(_build_base_model(cfg, dev)),
    ),
}


def _build_discrete_rnn(model_cfg: dict, device=None) -> UniPhyModel:
    from .components import apply_elman_rnn

    cfg = _normalize_cfg(model_cfg)
    model = _build_base_model(model_cfg, device=device)
    return apply_elman_rnn(model, int(cfg["embed_dim"]))


def build_variant(variant: str, model_cfg: dict, device=None):
    variant = str(variant).strip()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {sorted(VARIANTS)}")
    _, builder = VARIANTS[variant]
    return builder(model_cfg, device)


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
    from torch.optim import AdamW
    from Exp.ERA5.runtime_config import build_adamw_optimizer

    if variant == "A2_discrete_rnn":
        rnn_params, other_params = [], []
        rnn_names = {"rnn_h.weight", "rnn_u.weight", "rnn_u.bias"}
        for name, param in model.named_parameters():
            if any(rn in name for rn in rnn_names):
                rnn_params.append(param)
            else:
                other_params.append(param)
        return AdamW(
            [
                {"params": other_params, "lr": float(cfg["train"]["lr"])},
                {"params": rnn_params, "lr": float(cfg["train"]["lr"]) * 3.0},
            ],
            weight_decay=float(cfg["train"]["weight_decay"]),
        )
    return build_adamw_optimizer(model, cfg)
