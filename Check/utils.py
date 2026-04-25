from pathlib import Path

import torch

from Exp.ERA5.runtime_config import compute_channelwise_crps, compute_crps
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
        zero_mask = broadcast_mask(dt_value.abs() <= 1e-12, pred.ndim)
        pred = torch.where(zero_mask, x_curr, pred)
        x_curr = pred
        preds.append(pred)
    return torch.stack(preds, dim=1)


def manual_forward(model, x, dt):
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
            pairwise_total = pairwise_total + (
                pred_ensemble[left] - pred_ensemble[right]
            ).abs().mean()
    return mae - pairwise_total / (ensemble_size * ensemble_size)


def channelwise_crps_reference(pred_ensemble, target, lat_weights):
    ensemble_size = pred_ensemble.shape[0]
    mae = ((pred_ensemble - target.unsqueeze(0)).abs() * lat_weights).mean(
        dim=(-2, -1)
    ).mean(dim=0)
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
    actual = compute_crps(pred_ensemble, target)
    expected = crps_reference(pred_ensemble, target)
    return max_diff(actual, expected), actual, expected


def compute_channelwise_crps_error(pred_ensemble, target, lat_weights):
    actual = compute_channelwise_crps(pred_ensemble, target, lat_weights)
    expected = channelwise_crps_reference(pred_ensemble, target, lat_weights)
    return max_diff(actual, expected), actual, expected
