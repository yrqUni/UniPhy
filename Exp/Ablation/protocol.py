from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LEAD_TIMES = (6, 24, 72, 120, 240)
DEFAULT_PRIMARY_LEADS = (24, 120, 240)
DEFAULT_METRICS = ("rmse", "acc", "crps")
DEFAULT_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    group: str
    label: str
    factor: str
    intervention: str
    hypothesis: str
    expected_effect: str
    training_role: str = "model"

    def to_dict(self):
        return asdict(self)


VARIANT_SPECS = {
    "baseline": VariantSpec(
        name="baseline",
        group="Reference",
        label="Baseline",
        factor="Active UniPhy",
        intervention="No architectural or objective change.",
        hypothesis="The deterministic recurrent UniPhy core provides the reference skill under matched training and evaluation.",
        expected_effect="Strong aggregate RMSE and stable fixed interval rollout behavior.",
    ),
    "A1_no_dt": VariantSpec(
        name="A1_no_dt",
        group="A. Time conditioning",
        label="A1: no dt",
        factor="Explicit interval handling",
        intervention="Replace every observed time interval with dt_ref.",
        hypothesis="Explicit interval handling is needed when evaluation uses nonuniform lead grids.",
        expected_effect="Lower robustness on irregular time sensitivity grids.",
    ),
    "D1_single_scale": VariantSpec(
        name="D1_single_scale",
        group="D. Spatial coupling",
        label="D1: single scale",
        factor="Multi scale spatial mixing",
        intervention="Keep only the local spatial branch.",
        hypothesis="Multi scale receptive fields improve synoptic scale structure.",
        expected_effect="Higher RMSE on larger scale variables and longer leads.",
    ),
    "D2_fixed_scale_weights": VariantSpec(
        name="D2_fixed_scale_weights",
        group="D. Spatial coupling",
        label="D2: fixed scale weights",
        factor="Adaptive scale weighting",
        intervention="Fix the spatial scale gate to uniform logits.",
        hypothesis="Adaptive scale selection helps the model follow changing regimes.",
        expected_effect="Lower ACC and higher RMSE on variable regimes.",
    ),
    "E1_l1_only": VariantSpec(
        name="E1_l1_only",
        group="E. Training objective",
        label="E1: L1 only",
        factor="Objective composition",
        intervention="Train with L1 loss only.",
        hypothesis="The rollout objective is the primary determinant of recursive stability in the deterministic model.",
        expected_effect="Useful objective control under matched recurrent dynamics.",
        training_role="objective",
    ),
    "G1_swin_transformer": VariantSpec(
        name="G1_swin_transformer",
        group="G. Operational baselines",
        label="G1: SwinTrans",
        factor="Single frame transformer baseline",
        intervention="Replace UniPhy with a Swin style fixed interval single frame predictor.",
        hypothesis="A fixed interval transformer is competitive for one step prediction but does not provide native variable interval evaluation.",
        expected_effect="Competitive fixed step short horizon skill.",
    ),
    "G2_convlstm": VariantSpec(
        name="G2_convlstm",
        group="G. Operational baselines",
        label="G2: ConvLSTM",
        factor="Traditional recurrent convolutional baseline",
        intervention="Replace UniPhy with a ConvLSTM fixed interval recurrent predictor.",
        hypothesis="A recurrent convolutional model provides a classical fixed interval autoregressive baseline.",
        expected_effect="Useful fixed step reference with weaker medium range stability than UniPhy.",
    ),
}


VARIANT_ORDER = tuple(VARIANT_SPECS)


def get_variant_spec(name: str) -> VariantSpec:
    try:
        return VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation variant '{name}'") from exc


def variant_names(include_baseline: bool = True) -> list[str]:
    names = list(VARIANT_ORDER)
    if include_baseline:
        return names
    return [name for name in names if name != "baseline"]


def protocol_dict(
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    lead_times: Iterable[int] = DEFAULT_LEAD_TIMES,
    primary_leads: Iterable[int] = DEFAULT_PRIMARY_LEADS,
) -> dict:
    return {
        "design": "single-factor controlled ablation",
        "reference_variant": "baseline",
        "paired_seeds": [int(seed) for seed in seeds],
        "lead_times_hours": [int(lead) for lead in lead_times],
        "primary_leads_hours": [int(lead) for lead in primary_leads],
        "metrics": list(DEFAULT_METRICS),
        "selection_rule": (
            "Report all variants trained with identical data splits, optimizer, "
            "scheduler, lead times, and climatology. Compare each "
            "variant against the baseline at matched random seeds."
        ),
        "variants": [VARIANT_SPECS[name].to_dict() for name in VARIANT_ORDER],
    }


def build_run_manifest(
    *,
    variant: str,
    seed: int,
    cfg: dict,
    command: list[str] | None = None,
) -> dict:
    spec = get_variant_spec(variant)
    return {
        "variant": spec.to_dict(),
        "seed": int(seed),
        "config": cfg,
        "command": command or [],
        "protocol": protocol_dict(seeds=[seed]),
    }


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_result_file(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if "variant" not in obj and path.stem in VARIANT_SPECS:
        obj["variant"] = path.stem
    validate_result(obj, source=str(path))
    return obj


def load_results(results_dir: str | Path) -> list[dict]:
    skip_names = {
        "manifest.json",
        "protocol.json",
        "summary.json",
        "check_results.json",
    }
    return [
        load_result_file(path)
        for path in sorted(Path(results_dir).rglob("*.json"))
        if path.name not in skip_names
    ]


def validate_result(obj: dict, *, source: str = "<memory>") -> None:
    required = ("variant", "overall", "lead_times_hours", "num_samples")
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"{source}: missing required keys {missing}")
    variant = obj["variant"]
    if variant not in VARIANT_SPECS:
        raise ValueError(f"{source}: unknown variant '{variant}'")
    for metric in DEFAULT_METRICS:
        if metric not in obj["overall"]:
            raise ValueError(f"{source}: missing overall.{metric}")
        for lead in obj["lead_times_hours"]:
            if str(lead) not in obj["overall"][metric]:
                raise ValueError(f"{source}: missing {metric}@{lead}h")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def summarize_results(results: list[dict], lead_times: Iterable[int]) -> list[dict]:
    by_variant: dict[str, list[dict]] = {}
    for obj in results:
        by_variant.setdefault(obj["variant"], []).append(obj)

    baseline = by_variant.get("baseline", [])
    baseline_by_seed = {
        int(obj.get("seed", -1)): obj for obj in baseline if obj.get("seed") is not None
    }

    rows = []
    for variant in VARIANT_ORDER:
        entries = by_variant.get(variant, [])
        if not entries:
            continue
        spec = get_variant_spec(variant)
        row = {
            "variant": variant,
            "label": spec.label,
            "group": spec.group,
            "n": len(entries),
        }
        for metric in DEFAULT_METRICS:
            for lead in lead_times:
                key = f"{metric}@{int(lead)}h"
                vals = [
                    float(obj["overall"][metric][str(int(lead))])
                    for obj in entries
                    if str(int(lead)) in obj["overall"][metric]
                ]
                mean_val = _mean(vals)
                std_val = _std(vals)
                row[key] = mean_val
                row[f"{key}_std"] = std_val
                row[f"{key}_ci95"] = (
                    None
                    if mean_val is None or std_val is None
                    else 1.96 * std_val / math.sqrt(max(len(vals), 1))
                )

                paired = []
                for obj in entries:
                    seed = obj.get("seed")
                    if seed is None or int(seed) not in baseline_by_seed:
                        continue
                    base = baseline_by_seed[int(seed)]
                    if str(int(lead)) in base["overall"][metric]:
                        paired.append(
                            float(obj["overall"][metric][str(int(lead))])
                            - float(base["overall"][metric][str(int(lead))])
                        )
                row[f"{key}_delta_vs_baseline"] = _mean(paired)
        rows.append(row)
    return rows
