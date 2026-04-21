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


def _deprecated_module_names():
    return {
        "Riem" + "annian" + "Clif" + "fordConv2d",
        "Efficient" + "SpatialPool",
        "Multi" + "Slot" + "FluxTracker",
        "Grouped" + "Decoder" + "SkipGate",
        "Original" + "Decoder" + "SkipGate",
        "External" + "ContextPath",
        "DC" + "ConservationPath",
        "Temporal" + "ContextModulator",
    }


def _no_variant_attr_names():
    return [
        "variant" + "_config",
        "variant" + "_name",
        "_use_" + "efficient_spatial",
        "_use_" + "multi_slot_flux",
        "_use_" + "grouped_skip",
        "_use_" + "per_mode_alpha",
        "_remove_" + "metric",
        "decoder_skip_gate",
    ]


def _forbidden_constructor_params():
    return [
        "variant" + "_config",
        "variant",
        "use_" + "per_mode_alpha",
        "remove_" + "metric",
    ]


def check_no_variant_attributes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    present = []
    for prefix, module in [
        ("model", model),
        *[(f"block_{idx}", block) for idx, block in enumerate(model.blocks)],
    ]:
        for name in _no_variant_attr_names():
            if hasattr(module, name):
                present.append(f"{prefix}.{name}")
    passed = not present
    return passed, f"present={present}"


def check_no_deprecated_submodules():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    bad = []
    forbidden = _deprecated_module_names()
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in forbidden:
            bad.append(f"{name}:{cls_name}")
    passed = not bad
    return passed, f"modules={bad}"


def check_constructor_has_no_variant_param():
    names = inspect.signature(UniPhyModel.__init__).parameters.keys()
    bad = [name for name in _forbidden_constructor_params() if name in names]
    passed = not bad
    return passed, f"params={list(names)}"


def check_block_constructor_has_no_variant_param():
    names = inspect.signature(UniPhyBlock.__init__).parameters.keys()
    bad = [("variant" + "_config") for _ in [0] if ("variant" + "_config") in names]
    passed = not bad
    return passed, f"params={list(names)}"


def check_all_parameters_require_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device)
    frozen = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]
    passed = not frozen
    return passed, f"frozen={frozen}"


def check_no_linalg_ops():
    src = inspect.getsource(ComplexSVDTransform)
    has_solve = "linalg.solve" in src
    has_inv = "linalg.inv" in src
    passed = not has_solve and not has_inv
    detail = f"linalg.solve={has_solve} linalg.inv={has_inv}"
    return passed, detail


def check_no_legacy_basis_methods():
    has_biortho = hasattr(ComplexSVDTransform, "get_biorthogonal_pair")
    has_encode = hasattr(ComplexSVDTransform, "encode")
    has_decode = hasattr(ComplexSVDTransform, "decode")
    passed = not any([has_biortho, has_encode, has_decode])
    detail = (
        f"get_biorthogonal_pair={has_biortho} encode={has_encode} "
        f"decode={has_decode}"
    )
    return passed, detail


def check_no_combine_output_helper():
    src = inspect.getsource(UniPhyBlock)
    present = "_combine_output" in src
    return not present, f"_combine_output_present={present}"


def check_block_init_signature_clean():
    sig = str(inspect.signature(UniPhyBlock.__init__))
    has_img_h = "img_height" in sig
    has_img_w = "img_width" in sig
    has_kernel = "kernel_size" in sig
    passed = not any([has_img_h, has_img_w, has_kernel])
    detail = (
        f"img_height={has_img_h} img_width={has_img_w} " f"kernel_size={has_kernel}"
    )
    return passed, detail


TEST_ID = "S04"

CHECK_GROUPS = [
    (
        "Group 4: Architecture Verification",
        [
            check_no_variant_attributes,
            check_no_deprecated_submodules,
            check_constructor_has_no_variant_param,
            check_block_constructor_has_no_variant_param,
            check_all_parameters_require_grad,
            check_no_linalg_ops,
            check_no_legacy_basis_methods,
            check_no_combine_output_helper,
            check_block_init_signature_clean,
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
