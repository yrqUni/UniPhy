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
        factor="Full UniPhy",
        intervention="No architectural or objective change.",
        hypothesis="The full continuous-time stochastic dissipative model without residual readout injection provides the reference skill.",
        expected_effect="Best aggregate RMSE, ACC, and CRPS across long lead times.",
    ),
    "A1_no_dt": VariantSpec(
        name="A1_no_dt",
        group="A. Continuous-time sampling",
        label="A1: no continuous dt",
        factor="Explicit time interval conditioning",
        intervention="Replace every observed delta t with the reference interval dt_ref.",
        hypothesis="Irregular sampling requires explicit physical-time conditioning.",
        expected_effect="Largest degradation on non-uniform and long-horizon lead times.",
    ),
    "A2_discrete_rnn": VariantSpec(
        name="A2_discrete_rnn",
        group="A. Continuous-time sampling",
        label="A2: discrete RNN",
        factor="Matrix-exponential SDE transition",
        intervention="Replace the SDE propagator with a dt-agnostic Elman recurrence.",
        hypothesis="Closed-form continuous-time propagation improves extrapolation over a plain recurrence.",
        expected_effect="Lower ACC and higher RMSE at 120 h and 240 h.",
    ),
    "B1_complex_latent": VariantSpec(
        name="B1_complex_latent",
        group="B. Complex latent dynamics",
        label="B1: complex latent",
        factor="Dissipative real latent state",
        intervention="Use the previous complex eigenspace pathway in place of the dissipative real latent pathway.",
        hypothesis="A real dissipative latent state improves stability for finite-data medium-range training.",
        expected_effect="Higher long-horizon RMSE and CRPS under the complex pathway.",
    ),
    "B2_fixed_decay": VariantSpec(
        name="B2_fixed_decay",
        group="B. Latent memory",
        label="B2: fixed decay",
        factor="Learned continuous decay spectrum",
        intervention="Replace all learned decay rates with a fixed reference rate.",
        hypothesis="Channel-wise learned decay constants are necessary for multi-scale atmospheric memory.",
        expected_effect="Higher RMSE and lower ACC at 120 h and 240 h.",
    ),
    "H1_dt_relaxation_rnn": VariantSpec(
        name="H1_dt_relaxation_rnn",
        group="H. Candidate dynamics",
        label="H1: dt relaxation recurrence",
        factor="Low-variance physical-time recurrence",
        intervention="Replace stochastic latent propagation with a deterministic recurrence relaxed by 1 - exp(-lambda dt).",
        hypothesis="A low-variance recurrence with explicit physical-time relaxation preserves the short-horizon skill of discrete recurrence while retaining dt sensitivity.",
        expected_effect="Lower fixed-interval RMSE than stochastic UniPhy and better variable-dt behavior than dt-agnostic recurrence.",
    ),
    "C1_deterministic": VariantSpec(
        name="C1_deterministic",
        group="C. Probabilistic dynamics",
        label="C1: deterministic",
        factor="Stochastic forcing",
        intervention="Remove latent diffusion and evaluate a single deterministic member.",
        hypothesis="Stochastic forcing is required for calibrated probabilistic forecasts.",
        expected_effect="Higher CRPS and lower ensemble spread, with possible RMSE trade-offs.",
    ),
    "C2_no_readout_residual": VariantSpec(
        name="C2_no_readout_residual",
        group="C. Probabilistic dynamics",
        label="C2: residual readout",
        factor="Residual latent readout",
        intervention="Add a time-decayed residual convolutional readout inside each temporal block.",
        hypothesis="Residual readout injection can over-correct the dissipative latent trajectory during free forecasts.",
        expected_effect="Higher RMSE than the residual-free baseline, especially under variable dt.",
    ),
    "C3_constant_readout": VariantSpec(
        name="C3_constant_readout",
        group="C. Probabilistic dynamics",
        label="C3: constant readout",
        factor="Physical time scale of residual latent readout",
        intervention="Add a residual convolutional readout and keep it constant across lead time.",
        hypothesis="A persistent residual correction can destabilize free forecasts because unresolved fast closure should not remain constant as lead time grows.",
        expected_effect="Higher medium-range RMSE and CRPS under variable dt.",
    ),
    "D1_single_scale": VariantSpec(
        name="D1_single_scale",
        group="D. Spatial coupling",
        label="D1: single scale",
        factor="Multi-scale spatial mixing",
        intervention="Keep only the local 3x3 spatial branch.",
        hypothesis="Multi-scale receptive fields improve synoptic-scale structure.",
        expected_effect="Higher RMSE on large-scale variables and long lead times.",
    ),
    "D2_fixed_scale_weights": VariantSpec(
        name="D2_fixed_scale_weights",
        group="D. Spatial coupling",
        label="D2: fixed scale weights",
        factor="Adaptive multi-scale weighting",
        intervention="Fix the local, regional, and large-scale spatial branches to uniform weights.",
        hypothesis="Adaptive scale selection helps the model follow changing synoptic regimes.",
        expected_effect="Lower ACC and higher RMSE on long lead times.",
    ),
    "E1_l1_only": VariantSpec(
        name="E1_l1_only",
        group="E. Training objective",
        label="E1: L1 only",
        factor="CRPS ensemble objective",
        intervention="Train with L1 loss only and remove CRPS from the optimization objective.",
        hypothesis="CRPS improves probabilistic calibration without sacrificing point skill.",
        expected_effect="Higher CRPS and lower ensemble spread.",
        training_role="objective",
    ),
    "F1_etd1_integrator": VariantSpec(
        name="F1_etd1_integrator",
        group="F. Temporal integration",
        label="F1: Euler transition",
        factor="Exact dissipative transition",
        intervention="Replace the exact exponential dissipative update with a first-order Euler update.",
        hypothesis="Exact continuous-time propagation improves stability when inference uses variable dt.",
        expected_effect="Worse RMSE and ACC at long lead times, especially for variable dt.",
    ),
    "G1_swin_transformer": VariantSpec(
        name="G1_swin_transformer",
        group="G. Operational baselines",
        label="G1: SwinTrans",
        factor="Single-frame transformer baseline",
        intervention="Replace UniPhy with a Swin-style window-attention single-frame predictor.",
        hypothesis="A fixed-interval single-frame transformer is competitive for one-step prediction but lacks native continuous-time rollout.",
        expected_effect="Competitive fixed-step short-horizon skill with weaker variable-time generalization.",
    ),
    "G2_convlstm": VariantSpec(
        name="G2_convlstm",
        group="G. Operational baselines",
        label="G2: ConvLSTM",
        factor="Traditional recurrent convolutional baseline",
        intervention="Replace UniPhy with a ConvLSTM fixed-step recurrent predictor.",
        hypothesis="A recurrent convolutional model provides a classical fixed-interval autoregressive baseline.",
        expected_effect="Useful fixed-step reference with weaker medium-range stability than UniPhy.",
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
            "scheduler, ensemble size, lead times, and climatology. Compare each "
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
