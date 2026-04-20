import inspect
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import write_result
else:
    from ..utils import write_result

import torch

from Model.UniPhy import UniPhyOps
from Model.UniPhy.ModelUniPhy import UniPhyBlock, UniPhyModel
from Model.UniPhy.PScan import pscan
from Model.UniPhy.UniPhyOps import (
    ComplexSVDTransform,
    GlobalFluxTracker,
    MultiScaleSpatialMixer,
    TemporalPropagator,
    _compute_sde_scale,
    _safe_forcing,
)


def build_check_model(device, dtype=torch.float64):
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=64,
        expand=4,
        depth=2,
        patch_size=(4, 2),
        img_height=32,
        img_width=32,
        dt_ref=1.0,
        init_noise_scale=0.01,
    ).to(device)
    if dtype == torch.float64:
        model = model.double()
    elif dtype != torch.float32:
        raise ValueError(f"unsupported_dtype={dtype}")
    return model


def _forbidden_encoder_names():
    return ["metric" + "_weight", "rho", "metric" + "_det"]


def _deprecated_param_keywords():
    return [
        "cliff",
        "metric",
        "rho",
        "efficient_spatial",
        "multi_slot",
        "grouped_skip",
        "spectral_mod",
        "external_context",
        "dc_conservation",
        "temporal_context_mod",
    ]


def check_dt_ref_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    values = []
    for block in model.blocks:
        values.append(float(block.prop.dt_ref))
        if float(block.prop.dt_ref) != float(block.prop.flux_tracker.dt_ref):
            return False, "block_dt_ref_mismatch"
    passed = max(values) - min(values) == 0.0
    return passed, f"dt_ref_values={values}"


def check_noise_scale_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    values = torch.tensor(
        [float(block.prop.base_noise.abs().mean().item()) for block in model.blocks],
        dtype=torch.float64,
    )
    variance = float(values.var(unbiased=False).item()) if values.numel() > 1 else 0.0
    passed = variance < 1e-6
    return passed, f"variance={variance:.3e} values={values.tolist()}"


def check_state_dimension_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    states = model._init_states(3, device, torch.complex64)
    expected_h = (3 * model.h_patches * model.w_patches, 1, model.embed_dim)
    expected_flux = (3, model.blocks[0].prop.flux_tracker.state_dim)
    for block_idx, (h_state, flux_state) in enumerate(states):
        if tuple(h_state.shape) != expected_h:
            return False, f"block={block_idx} h_shape={tuple(h_state.shape)}"
        if tuple(flux_state.shape) != expected_flux:
            return False, f"block={block_idx} flux_shape={tuple(flux_state.shape)}"
    return True, f"h_shape={expected_h} flux_shape={expected_flux}"


def check_no_duplicate_parameter_names():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    names = [name for name, _ in model.named_parameters()]
    unique = len(set(names))
    passed = unique == len(names)
    return passed, f"count={len(names)} unique={unique}"


def check_no_dead_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for name, _ in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in _deprecated_param_keywords()):
            bad.append(name)
    passed = not bad
    return passed, f"bad_names={bad}"


def check_basis_is_scalar_alpha():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for block_idx, block in enumerate(model.blocks):
        basis = block.prop.basis
        if basis.alpha_logit.shape != torch.Size([]):
            bad.append(
                f"block={block_idx} alpha_shape={tuple(basis.alpha_logit.shape)}"
            )
        if (
            hasattr(basis, "spectral_mod")
            and getattr(basis, "spectral_mod") is not None
        ):
            bad.append(f"block={block_idx} spectral_mod_present")
    passed = not bad
    return passed, f"issues={bad}"


def check_single_spatial_mixer_type():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    for block_idx, block in enumerate(model.blocks):
        if not isinstance(block.spatial_mixer, MultiScaleSpatialMixer):
            bad.append(
                f"block={block_idx} wrong_type={type(block.spatial_mixer).__name__}"
            )
        for name in ["spatial_cliff", "spatial_pool", "norm_spatial"]:
            if hasattr(block, name):
                bad.append(f"block={block_idx} has_{name}")
    passed = not bad
    return passed, f"issues={bad}"


def check_single_flux_tracker_type():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    extra_name = "Multi" + "Slot" + "FluxTracker"
    for block_idx, block in enumerate(model.blocks):
        if not isinstance(block.prop.flux_tracker, GlobalFluxTracker):
            bad.append(
                "block="
                f"{block_idx} wrong_flux="
                f"{type(block.prop.flux_tracker).__name__}"
            )
    extra = [
        type(module).__name__
        for _, module in model.named_modules()
        if type(module).__name__ == extra_name
    ]
    passed = not bad and not extra
    return passed, f"issues={bad + extra}"


def check_encoder_has_no_metric():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    present = [
        name for name in _forbidden_encoder_names() if hasattr(model.encoder, name)
    ]
    passed = not present
    return passed, f"present={present}"


def check_skip_is_inlined():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    has_required = hasattr(model, "skip_context_proj") and hasattr(
        model, "skip_spatial_proj"
    )
    has_wrapper = hasattr(model, "decoder_skip_gate")
    passed = has_required and not has_wrapper
    return passed, f"has_required={has_required} has_wrapper={has_wrapper}"


def check_numerical_constants():
    src_ops = inspect.getsource(UniPhyOps)
    checks = {
        "eps_1e7_present": "eps=1e-7" in src_ops,
        "alpha_logit_init_2": "tensor(2.0)" in src_ops,
        "gate_min_001": "0.01" in src_ops,
        "gate_max_099": "0.99" in src_ops,
    }
    passed = all(checks.values())
    detail = " ".join(f"{name}={value}" for name, value in checks.items())
    return passed, detail


TEST_ID = "S03"

CHECK_GROUPS = [
    (
        "Group 3: Parameter Consistency",
        [
            check_dt_ref_consistency,
            check_noise_scale_consistency,
            check_state_dimension_consistency,
            check_no_duplicate_parameter_names,
            check_no_dead_parameters,
            check_basis_is_scalar_alpha,
            check_single_spatial_mixer_type,
            check_single_flux_tracker_type,
            check_encoder_has_no_metric,
            check_skip_is_inlined,
            check_numerical_constants,
        ],
    )
]


def run():
    total = 0
    pass_count = 0
    overall_passed = True
    print("=" * 72)
    for current_group, checks in CHECK_GROUPS:
        print(current_group)
        print("-" * 72)
        group_pass_count = 0
        for check_fn in checks:
            total += 1
            try:
                passed, detail = check_fn()
            except Exception as exc:
                passed = False
                detail = f"{type(exc).__name__}: {exc}"
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {check_fn.__name__} :: {detail}")
            if passed:
                pass_count += 1
                group_pass_count += 1
            else:
                overall_passed = False
        print(f"GROUP SUMMARY {group_pass_count}/{len(checks)}")
        print("-" * 72)
    status = "PASS" if overall_passed else "FAIL"
    print("=" * 72)
    print(f"TOTAL {pass_count}/{total}")
    print(f"RESULT {status}")
    print("=" * 72)
    return status, pass_count, total


if __name__ == "__main__":
    status, pass_count, total = run()
    detail = f"pass_count={pass_count} total={total}"
    write_result(TEST_ID, status, "-", detail)
    sys.exit(0 if status == "PASS" else 1)
