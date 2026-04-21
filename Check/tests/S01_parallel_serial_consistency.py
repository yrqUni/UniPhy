import inspect
import sys
from pathlib import Path

from Check.utils import write_result

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


def _broadcast_zero_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


def _manual_forward(model, x, dt):
    batch_size, steps = x.shape[:2]
    device = x.device
    dt_seq = model._normalize_dt(dt, batch_size, steps, device)
    z_all = model.encoder(x)
    states = model._init_states(batch_size, device, z_all.dtype)
    outputs = []

    for step_idx in range(steps):
        z_step = z_all[:, step_idx]
        z_skip = z_all[:, step_idx]
        dt_step = dt_seq[:, step_idx]
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_step, h_next, flux_next = block.forward_step(
                z_step,
                h_prev,
                dt_step,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        out_step = model.decoder(
            model._apply_decoder_skip(
                z_step.unsqueeze(1),
                z_skip.unsqueeze(1),
            )
        )[:, 0]
        zero_mask = _broadcast_zero_mask(dt_step.abs() <= 1e-12, out_step.ndim)
        out_step = torch.where(zero_mask, x[:, step_idx], out_step)
        outputs.append(out_step)

    return torch.stack(outputs, dim=1)


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
        dt_value = model._normalize_step_dt(dt_step, batch_size, device)
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_curr, h_next, flux_next = block.forward_step(
                z_curr,
                h_prev,
                dt_value,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        pred = model.decoder(
            model._apply_decoder_skip(
                z_curr.unsqueeze(1),
                z_skip.unsqueeze(1),
            )
        )[:, 0]
        zero_mask = _broadcast_zero_mask(dt_value.abs() <= 1e-12, pred.ndim)
        pred = torch.where(zero_mask, x_curr, pred)
        x_curr = pred
        preds.append(pred)
    return torch.stack(preds, dim=1)


def check_parallel_vs_serial_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)
    model = build_check_model(device, torch.float64)
    model.eval()
    x = _make_input(2, 5, 4, 32, 32, device, torch.float64)
    dt = _make_dt(2, 5, device, torch.float64)
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_serial = _manual_forward(model, x, dt)
    diff = _max_diff(out_parallel, out_serial)
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"


def check_rollout_vs_serial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12)
    model = build_check_model(device, torch.float64)
    model.eval()
    x_context = _make_input(2, 4, 4, 32, 32, device, torch.float64)
    dt_context = torch.tensor(
        [[1.0, 1.25, 0.5, 0.75], [1.0, 0.75, 1.5, 1.25]],
        device=device,
        dtype=torch.float64,
    )
    dt_list = [
        torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
        torch.tensor([0.25, 0.75], device=device, dtype=torch.float64),
        torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
    ]
    with torch.no_grad():
        rollout = model.forward_rollout(x_context, dt_context, dt_list)
        serial = full_serial_inference(model, x_context, dt_context, dt_list)
    diff = _max_diff(rollout, serial)
    passed = diff < 1e-7
    return passed, f"max_diff={diff:.3e}"


def check_flux_scan_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(13)
    batch_size, steps, dim = 2, 7, 16
    tracker = GlobalFluxTracker(dim=dim, dt_ref=1.0).to(device).double()
    x_mean_seq = torch.complex(
        torch.randn(batch_size, steps, dim, device=device, dtype=torch.float64),
        torch.randn(batch_size, steps, dim, device=device, dtype=torch.float64),
    )
    dt_seq = (
        torch.rand(batch_size, steps, device=device, dtype=torch.float64) * 1.5 + 0.1
    )
    flux0 = tracker.get_initial_state(batch_size, device, torch.complex128)
    scan_a, scan_x = tracker.get_scan_operators(x_mean_seq, dt_seq)
    flux_scan = pscan(scan_a, scan_x).squeeze(-1)
    flux_scan = flux_scan + flux0.unsqueeze(1) * torch.cumprod(
        scan_a.squeeze(-1), dim=1
    )

    flux_serial = []
    flux_state = flux0
    for step_idx in range(steps):
        flux_state, _, _ = tracker.forward_step(
            flux_state,
            x_mean_seq[:, step_idx],
            dt_seq[:, step_idx],
        )
        flux_serial.append(flux_state)
    flux_serial = torch.stack(flux_serial, dim=1)
    diff = _max_diff(flux_scan, flux_serial)
    passed = diff < 1e-10
    return passed, f"max_diff={diff:.3e}"


def check_temporal_propagator_scan_vs_serial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(14)
    prop = (
        TemporalPropagator(dim=32, dt_ref=1.0, init_noise_scale=0.01)
        .to(device)
        .double()
    )
    dt_seq = torch.rand(3, 5, device=device, dtype=torch.float64) * 1.5 + 0.1
    decay_seq, forcing_seq = prop.get_transition_operators_seq(dt_seq)
    max_decay = 0.0
    max_forcing = 0.0
    for step_idx in range(dt_seq.shape[1]):
        decay_step, forcing_step = prop.get_transition_operators_step(
            dt_seq[:, step_idx]
        )
        max_decay = max(max_decay, _max_diff(decay_seq[:, step_idx], decay_step))
        max_forcing = max(
            max_forcing, _max_diff(forcing_seq[:, step_idx], forcing_step)
        )
    diff = max(max_decay, max_forcing)
    passed = diff < 1e-12
    return passed, f"max_diff={diff:.3e}"


def check_multi_scale_mixer_real_imag_independence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(15)
    mixer = MultiScaleSpatialMixer(dim=8).to(device).double()
    real = torch.randn(2, 8, 9, 11, device=device, dtype=torch.float64)
    imag = torch.randn(2, 8, 9, 11, device=device, dtype=torch.float64)
    x = torch.complex(real, imag)
    with torch.no_grad():
        out = mixer(x)
        expected_real = real + mixer._forward_real(real) * mixer.output_scale.to(
            real.dtype
        )
        expected_imag = imag + mixer._forward_real(imag) * mixer.output_scale.to(
            imag.dtype
        )
        expected = torch.complex(expected_real, expected_imag)
    diff = _max_diff(out, expected)
    passed = diff < 1e-14
    return passed, f"max_diff={diff:.3e}"


TEST_ID = "S01"

CHECK_GROUPS = [
    (
        "Group 1: Parallel-Serial Consistency",
        [
            check_parallel_vs_serial_forward,
            check_rollout_vs_serial,
            check_flux_scan_equivalence,
            check_temporal_propagator_scan_vs_serial,
            check_multi_scale_mixer_real_imag_independence,
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
