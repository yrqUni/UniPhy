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


def _make_input(batch_size, steps, channels, height, width, device, dtype):
    return torch.randn(
        batch_size,
        steps,
        channels,
        height,
        width,
        device=device,
        dtype=dtype,
    )


def _make_dt(batch_size, steps, device, dtype):
    base = torch.tensor(
        [0.5, 1.0, 1.25, 0.75, 1.5, 0.6],
        device=device,
        dtype=dtype,
    )
    repeats = (steps + base.numel() - 1) // base.numel()
    seq = base.repeat(repeats)[:steps]
    return seq.unsqueeze(0).repeat(batch_size, 1).contiguous()


def _max_diff(a, b):
    return float((a - b).abs().max().item())


def _mean_diff(a, b):
    return float((a - b).abs().mean().item())


def _broadcast_zero_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


def full_serial_inference(model, x_context, dt_context, dt_list):
    batch_size, context_steps = x_context.shape[:2]
    device = x_context.device
    z_all = model.encoder(x_context)
    z_skip = z_all[:, -1]
    states = model._init_states(batch_size, device, z_all.dtype)
    dt_ctx = model._normalize_dt(dt_context, batch_size, context_steps, device)

    if context_steps > 1:
        for step_idx in range(context_steps - 1):
            z_step = z_all[:, step_idx]
            dt_step = dt_ctx[:, step_idx + 1]
            for block_idx, block in enumerate(model.blocks):
                h_prev, flux_prev = states[block_idx]
                z_step, h_next, flux_next = block.forward_step(
                    z_step,
                    h_prev,
                    dt_step,
                    flux_prev,
                )
                states[block_idx] = (h_next, flux_next)

    z_curr = z_all[:, -1]
    x_curr = x_context[:, -1]
    preds = []
    for dt_step in dt_list:
        dt_value = model._normalize_dt(dt_step, batch_size, 1, device).squeeze(1)
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_curr, h_next, flux_next = block.forward_step(
                z_curr,
                h_prev,
                dt_value,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        pred = model.decoder(model._apply_decoder_skip(z_curr, z_skip))
        zero_mask = _broadcast_zero_mask(dt_value.abs() <= 1e-12, pred.ndim)
        pred = torch.where(zero_mask, x_curr, pred)
        x_curr = pred
        preds.append(pred)
    return torch.stack(preds, dim=1)


def check_dt_zero_is_identity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(21)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt = torch.zeros(2, 4, device=device, dtype=torch.float64)
    with torch.no_grad():
        out = model(x, dt)
    diff = _max_diff(out, x)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"


def check_dt_scaling_changes_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(22)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_small = torch.full((2, 4), 0.5, device=device, dtype=torch.float64)
    dt_large = torch.full((2, 4), 2.0, device=device, dtype=torch.float64)
    with torch.no_grad():
        out_small = model(x, dt_small)
        out_large = model(x, dt_large)
    diff = _mean_diff(out_small, out_large)
    passed = diff > 1e-6
    return passed, f"mean_diff={diff:.3e}"


def check_rollout_horizon_semantics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(23)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = torch.tensor(
        [[1.0, 1.25, 0.5, 0.75], [1.0, 0.75, 1.5, 1.25]],
        device=device,
        dtype=torch.float64,
    )
    dt_first = torch.tensor([0.5, 1.0], device=device, dtype=torch.float64)
    dt_list = [dt_first, dt_first * 0.5, dt_first * 1.5]
    with torch.no_grad():
        rollout = model.forward_rollout(x_context, dt_context, dt_list)
        first_pred = rollout[:, 0]
        serial = full_serial_inference(model, x_context, dt_context, dt_list)
    diff = _max_diff(first_pred, serial[:, 0])
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"


def check_rollout_stride_offset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(24)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = _make_dt(2, 4, device, torch.float64)
    dt_list = [
        torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
        torch.tensor([0.75, 1.25], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        rollout_full = model.forward_rollout(x_context, dt_context, dt_list)
        rollout_stride = model.forward_rollout(
            x_context,
            dt_context,
            dt_list,
            output_stride=2,
            output_offset=1,
        )
    diff = _max_diff(rollout_stride, rollout_full[:, 1::2])
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"


def check_context_dt_scalar_vs_tensor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(25)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 3, 4, 32, 32, device, torch.float64)
    dt_tensor = torch.ones(2, 3, device=device, dtype=torch.float64)
    dt_list = [
        torch.tensor([0.5, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.25, 0.5], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        out_scalar = model.forward_rollout(x_context, 1.0, dt_list)
        out_tensor = model.forward_rollout(x_context, dt_tensor, dt_list)
    diff = _max_diff(out_scalar, out_tensor)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"


def check_small_eigenvalue_limits():
    dt_ratio = torch.tensor([0.25, 1.5], dtype=torch.float64)
    exp_arg = torch.zeros(2, dtype=torch.complex128)
    forcing = _safe_forcing(exp_arg, dt_ratio)
    forcing_diff = float((forcing.real - dt_ratio).abs().max().item())
    sde_scale = _compute_sde_scale(
        torch.zeros(2, dtype=torch.float64),
        dt_ratio,
        torch.ones(2, dtype=torch.float64),
    )
    sde_diff = float((sde_scale - torch.sqrt(dt_ratio)).abs().max().item())
    diff = max(forcing_diff, sde_diff)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"


def check_negative_dt_rejected():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(26)
    model = build_check_model(device, torch.float64)
    x = _make_input(2, 3, 4, 32, 32, device, torch.float64)
    try:
        model(x, -1.0)
    except ValueError as exc:
        return True, f"raised={type(exc).__name__}"
    return False, "negative_dt_not_rejected"


def check_dt_normalize_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_check_model(device, torch.float64)
    batch_size = 3
    steps = 4
    scalar = model._normalize_dt(1.0, batch_size, steps, device)
    zero_dim = model._normalize_dt(
        torch.tensor(2.0, device=device, dtype=torch.float64), batch_size, steps, device
    )
    batch_vec = model._normalize_dt(
        torch.tensor([0.5, 1.0, 1.5], device=device, dtype=torch.float64),
        batch_size,
        steps,
        device,
    )
    full = model._normalize_dt(
        torch.arange(batch_size * steps, device=device, dtype=torch.float64).reshape(
            batch_size, steps
        ),
        batch_size,
        steps,
        device,
    )
    shapes_ok = all(
        tensor.shape == (batch_size, steps)
        for tensor in [scalar, zero_dim, batch_vec, full]
    )
    values_ok = (
        torch.allclose(
            scalar,
            torch.full((batch_size, steps), 1.0, device=device, dtype=torch.float64),
        )
        and torch.allclose(
            zero_dim,
            torch.full((batch_size, steps), 2.0, device=device, dtype=torch.float64),
        )
        and torch.allclose(
            batch_vec[:, 0],
            torch.tensor([0.5, 1.0, 1.5], device=device, dtype=torch.float64),
        )
        and torch.allclose(
            full,
            torch.arange(
                batch_size * steps, device=device, dtype=torch.float64
            ).reshape(batch_size, steps),
        )
    )
    passed = shapes_ok and values_ok
    return passed, f"shapes_ok={shapes_ok} values_ok={values_ok}"


TEST_ID = "S02"

CHECK_GROUPS = [
    (
        "Group 2: Time-Step Semantics",
        [
            check_dt_zero_is_identity,
            check_dt_scaling_changes_output,
            check_rollout_horizon_semantics,
            check_rollout_stride_offset,
            check_context_dt_scalar_vs_tensor,
            check_small_eigenvalue_limits,
            check_negative_dt_rejected,
            check_dt_normalize_shapes,
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
