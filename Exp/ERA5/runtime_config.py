import copy
import os
from typing import Dict, List, Optional

import yaml

SURFACE_VARS = ["TCWV", "U10", "V10", "T2", "MSLP", "SP"]
PRESSURE_BASE_VARS = ["VV", "U", "V", "RH", "T", "Z"]
PRESSURE_LEVELS = ["925", "850", "500", "100"]
CHANNEL_NAMES = SURFACE_VARS + [
    f"{var}{level}" for var in PRESSURE_BASE_VARS for level in PRESSURE_LEVELS
]

DEFAULT_TRAIN_YEAR_RANGE = [2000, 2008]
DEFAULT_TEST_YEAR_RANGE = [2009, 2009]


def load_yaml_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_year_range(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    cleaned = text.replace(":", ",").strip()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        value = int(parts[0])
        return [value, value]
    if len(parts) != 2:
        raise ValueError(f"Invalid year range: {text}")
    return [int(parts[0]), int(parts[1])]


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if not text:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def resolve_data_input_dir(
    cfg: Dict[str, object],
    override: Optional[str] = None,
) -> str:
    if override:
        return override
    env_value = os.environ.get("UNIPHY_ERA5_DATA")
    if env_value:
        return env_value
    return str(cfg["data"]["input_dir"])


def resolve_train_year_range(
    cfg: Dict[str, object],
    override: Optional[str] = None,
) -> List[int]:
    parsed = parse_year_range(override)
    if parsed is not None:
        return parsed
    env_parsed = parse_year_range(os.environ.get("UNIPHY_TRAIN_YEAR_RANGE"))
    if env_parsed is not None:
        return env_parsed
    cfg_years = cfg["data"].get("year_range")
    if cfg_years:
        return [int(cfg_years[0]), int(cfg_years[1])]
    return list(DEFAULT_TRAIN_YEAR_RANGE)


def resolve_eval_year_range(override: Optional[str] = None) -> List[int]:
    parsed = parse_year_range(override)
    if parsed is not None:
        return parsed
    env_parsed = parse_year_range(os.environ.get("UNIPHY_EVAL_YEAR_RANGE"))
    if env_parsed is not None:
        return env_parsed
    return list(DEFAULT_TEST_YEAR_RANGE)


def build_runtime_cfg(
    config_path: str,
    *,
    data_input_dir: Optional[str] = None,
    train_year_range: Optional[str] = None,
    sample_offsets_hours: Optional[str] = None,
    epochs: Optional[int] = None,
    log_path: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    pretrained_ckpt: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    cfg = copy.deepcopy(load_yaml_config(config_path))
    cfg.setdefault("logging", {})
    cfg.setdefault("data", {})
    cfg.setdefault("runtime", {})
    cfg["data"]["input_dir"] = resolve_data_input_dir(cfg, data_input_dir)
    cfg["data"]["year_range"] = resolve_train_year_range(cfg, train_year_range)

    offsets = parse_float_list(sample_offsets_hours)
    if offsets is not None:
        cfg["data"]["sample_offsets_hours"] = offsets

    if epochs is not None:
        cfg["train"]["epochs"] = int(epochs)
    if log_path is not None:
        cfg["logging"]["log_path"] = log_path
    if ckpt_dir is not None:
        cfg["logging"]["ckpt_dir"] = ckpt_dir
    if ckpt_path is not None:
        cfg["logging"]["ckpt"] = ckpt_path
    if pretrained_ckpt is not None:
        cfg.setdefault("alignment", {})
        cfg["alignment"]["pretrained_ckpt"] = pretrained_ckpt
    cfg["runtime"]["max_steps"] = None if max_steps is None else int(max_steps)
    return cfg
