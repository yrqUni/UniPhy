from pathlib import Path

import torch

from Model.UniPhy.ModelUniPhy import UniPhyModel

REPO_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = str(REPO_DIR / "Check" / "logs")


def format_max_error(value):
    if value in {None, "-"}:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.6e}"


def write_result(test_id, status, max_error, detail, log_dir=LOG_DIR):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    payload = (
        "\n".join(
            [
                f"STATUS: {status}",
                f"MAX_ERROR: {format_max_error(max_error)}",
                f"TEST: {test_id}",
                f"DETAIL: {detail}",
            ]
        )
        + "\n"
    )
    path = Path(log_dir) / f"{test_id}_result.txt"
    path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return path


def build_check_model(device, dtype=torch.float64, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        return model.double()
    if dtype != torch.float32:
        raise ValueError(f"unsupported_dtype={dtype}")
    return model


def complex_randn(shape, device, dtype=torch.complex64):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, device=device, dtype=real_dtype)
    imag = torch.randn(shape, device=device, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype)


def inverse_softplus(value):
    value_tensor = torch.as_tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(value_tensor))


def fit_log_slope(xs, ys):
    x = torch.log(torch.as_tensor(xs, dtype=torch.float64))
    y = torch.log(torch.clamp(torch.as_tensor(ys, dtype=torch.float64), min=1e-30))
    x_centered = x - x.mean()
    denom = torch.sum(x_centered * x_centered)
    if float(denom.item()) == 0.0:
        return 0.0
    slope = torch.sum(x_centered * (y - y.mean())) / denom
    return float(slope.item())


def make_input(batch_size, steps, channels, height, width, device, dtype):
    return torch.randn(
        batch_size,
        steps,
        channels,
        height,
        width,
        device=device,
        dtype=dtype,
    )


def make_dt(batch_size, steps, device, dtype):
    base = torch.tensor(
        [0.5, 1.0, 1.25, 0.75, 1.5, 0.6],
        device=device,
        dtype=dtype,
    )
    repeats = (steps + base.numel() - 1) // base.numel()
    seq = base.repeat(repeats)[:steps]
    return seq.unsqueeze(0).repeat(batch_size, 1).contiguous()


def max_diff(a, b):
    return float((a - b).abs().max().item())


def broadcast_mask(mask, target_ndim):
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return mask


def full_serial_inference(model, x_context, dt_context, dt_list):
    batch_size, context_steps = x_context.shape[:2]
    device = x_context.device
    z_all = model.encoder(x_context)
    states = model._init_states(batch_size, device, z_all.dtype)
    dt_ctx = model._normalize_dt(dt_context, batch_size, context_steps, device)
    dt_steps_list = [
        model._normalize_step_dt(dt_step, batch_size, device) for dt_step in dt_list
    ]
    n_rollout = len(dt_steps_list)
    if n_rollout > 0:
        dt_next_list = dt_steps_list[1:] + [dt_steps_list[-1]]
    else:
        dt_next_list = []

    if context_steps >= 2:
        for step_idx in range(context_steps - 1):
            z_curr_for_block = z_all[:, step_idx]
            z_next_for_block = z_all[:, step_idx + 1]
            dt_step = dt_ctx[:, step_idx + 1]
            if step_idx + 2 < context_steps:
                dt_next = dt_ctx[:, step_idx + 2]
            elif len(dt_steps_list) > 0:
                dt_next = dt_steps_list[0]
            else:
                dt_next = dt_step
            for block_idx, block in enumerate(model.blocks):
                h_prev, flux_prev = states[block_idx]
                z_curr_for_block, h_next, flux_next = block.forward_step(
                    z_curr_for_block,
                    z_next_for_block,
                    h_prev,
                    dt_step,
                    dt_next,
                    flux_prev,
                    lead_time=dt_step,
                )
                z_next_for_block = z_curr_for_block
                states[block_idx] = (h_next, flux_next)

    z_curr = z_all[:, -1]
    x_curr = x_context[:, -1]
    preds = []
    lead_accum = torch.zeros_like(dt_steps_list[0]) if dt_steps_list else None
    for k, dt_value in enumerate(dt_steps_list):
        dt_next = dt_next_list[k]
        lead_accum = lead_accum + dt_value
        step_skip = z_curr

        z_pred_running = z_curr
        for i, block in enumerate(model.blocks):
            h_prev_i, flux_prev_i = states[i]
            z_pred_running, _hp, _fp = block.forward_step(
                z_pred_running,
                z_pred_running,
                h_prev_i,
                dt_value,
                dt_next,
                flux_prev_i,
                lead_time=lead_accum,
            )
        z_pred = z_pred_running

        new_states = []
        z_running = z_curr
        for i, block in enumerate(model.blocks):
            z_next_for_block = z_pred if i == 0 else z_running
            h_prev_i, flux_prev_i = states[i]
            z_running, h_next, flux_next = block.forward_step(
                z_running,
                z_next_for_block,
                h_prev_i,
                dt_value,
                dt_next,
                flux_prev_i,
                lead_time=lead_accum,
            )
            new_states.append((h_next, flux_next))
        states = new_states
        z_curr = model._apply_decoder_skip(
            z_running.unsqueeze(1),
            step_skip.unsqueeze(1),
        )[:, 0]
        pred = model.decoder(z_curr.unsqueeze(1))[:, 0]
        zero_mask = broadcast_mask(dt_value.abs() <= 1e-12, pred.ndim)
        pred = torch.where(zero_mask, x_curr, pred)
        x_curr = pred
        preds.append(pred)
    return torch.stack(preds, dim=1)


def manual_forward(model, x, dt):
    batch_size, steps = x.shape[:2]
    if steps < 2:
        raise ValueError(
            "manual_forward requires steps >= 2 to match the seq forward "
            "Cox-Matthews ETD2 contract (each step uses x_curr and x_next)."
        )
    device = x.device
    dt_seq = model._normalize_dt(dt, batch_size, steps, device)
    z_all = model.encoder(x)
    states = model._init_states(batch_size, device, z_all.dtype)
    outputs = []

    for step_idx in range(steps):
        z_skip = z_all[:, step_idx]
        dt_step = dt_seq[:, step_idx]
        if step_idx + 1 < steps:
            dt_next = dt_seq[:, step_idx + 1]
        else:
            dt_next = dt_seq[:, step_idx]
        z_curr_for_block = z_all[:, step_idx]
        if step_idx + 1 < steps:
            z_next_for_block = z_all[:, step_idx + 1]
        else:
            z_next_for_block = 2.0 * z_all[:, step_idx] - z_all[:, step_idx - 1]
        for block_idx, block in enumerate(model.blocks):
            h_prev, flux_prev = states[block_idx]
            z_curr_for_block, h_next, flux_next = block.forward_step(
                z_curr_for_block,
                z_next_for_block,
                h_prev,
                dt_step,
                dt_next,
                flux_prev,
            )
            states[block_idx] = (h_next, flux_next)
        out_step = model.decoder(
            model._apply_decoder_skip(
                z_curr_for_block.unsqueeze(1),
                z_skip.unsqueeze(1),
            )
        )[:, 0]
        zero_mask = broadcast_mask(dt_step.abs() <= 1e-12, out_step.ndim)
        out_step = torch.where(zero_mask, x[:, step_idx], out_step)
        outputs.append(out_step)

    return torch.stack(outputs, dim=1)


def crps_reference(pred_ensemble, target):
    ensemble_size = pred_ensemble.shape[0]
    mae = (pred_ensemble - target.unsqueeze(0)).abs().mean()
    if ensemble_size <= 1:
        return mae
    pairwise_total = torch.tensor(0.0, device=pred_ensemble.device, dtype=mae.dtype)
    for left in range(ensemble_size):
        for right in range(left + 1, ensemble_size):
            pairwise_total = (
                pairwise_total
                + (pred_ensemble[left] - pred_ensemble[right]).abs().mean()
            )
    return mae - pairwise_total / (ensemble_size * ensemble_size)


def channelwise_crps_reference(pred_ensemble, target, lat_weights):
    ensemble_size = pred_ensemble.shape[0]
    mae = (
        ((pred_ensemble - target.unsqueeze(0)).abs() * lat_weights)
        .mean(dim=(-2, -1))
        .mean(dim=0)
    )
    if ensemble_size <= 1:
        return mae
    pairwise_total = torch.zeros_like(mae)
    for left in range(ensemble_size):
        for right in range(left + 1, ensemble_size):
            pairwise_total = pairwise_total + (
                (pred_ensemble[left] - pred_ensemble[right]).abs() * lat_weights
            ).mean(dim=(-2, -1))
    return mae - pairwise_total / (ensemble_size * ensemble_size)


def compute_crps_error(pred_ensemble, target):
    ensemble_size = pred_ensemble.shape[0]
    mae = (pred_ensemble - target.unsqueeze(0)).abs().mean()
    if ensemble_size <= 1:
        actual = mae
    else:
        idx_i, idx_j = torch.triu_indices(
            ensemble_size,
            ensemble_size,
            offset=1,
            device=pred_ensemble.device,
        )
        scale = float(idx_i.shape[0]) / float(ensemble_size * ensemble_size)
        pairwise = (pred_ensemble[idx_i] - pred_ensemble[idx_j]).abs().mean()
        actual = mae - pairwise * scale
    expected = crps_reference(pred_ensemble, target)
    return max_diff(actual, expected), actual, expected


def compute_channelwise_crps_error(pred_ensemble, target, lat_weights):
    ensemble_size = pred_ensemble.shape[0]
    mae = (
        ((pred_ensemble - target.unsqueeze(0)).abs() * lat_weights)
        .mean(dim=(-2, -1))
        .mean(dim=0)
    )
    if ensemble_size <= 1:
        actual = mae
    else:
        idx_i, idx_j = torch.triu_indices(
            ensemble_size,
            ensemble_size,
            offset=1,
            device=pred_ensemble.device,
        )
        scale = float(idx_i.shape[0]) / float(ensemble_size * ensemble_size)
        pairwise = (
            ((pred_ensemble[idx_i] - pred_ensemble[idx_j]).abs() * lat_weights)
            .mean(dim=(-2, -1))
            .mean(dim=0)
        )
        actual = mae - pairwise * scale
    expected = channelwise_crps_reference(pred_ensemble, target, lat_weights)
    return max_diff(actual, expected), actual, expected
