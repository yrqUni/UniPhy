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
    payload = "\n".join(
        [
            f"STATUS: {status}",
            f"MAX_ERROR: {format_max_error(max_error)}",
            f"TEST: {test_id}",
            f"DETAIL: {detail}",
        ]
    ) + "\n"
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
        embed_dim=16,
        expand=2,
        depth=2,
        patch_size=(4, 4),
        img_height=16,
        img_width=16,
        dt_ref=1.0,
    ).to(device)
    if dtype == torch.float64:
        return model.double()
    if dtype != torch.float32:
        raise ValueError(f"unsupported_dtype={dtype}")
    return model


def make_input(batch_size, steps, channels, height, width, device, dtype):
    return torch.randn(batch_size, steps, channels, height, width, device=device, dtype=dtype)


def make_dt(batch_size, steps, device, dtype):
    base = torch.tensor([0.5, 1.0, 1.5, 0.75, 2.0, 1.25], device=device, dtype=dtype)
    repeats = (steps + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:steps].unsqueeze(0).repeat(batch_size, 1).contiguous()


def max_diff(a, b):
    return float((a - b).abs().max().item())


def serial_diag_scan(a, x):
    squeeze = x.ndim == 4
    if squeeze:
        x = x.unsqueeze(-1)
    y = []
    state = torch.zeros_like(x[:, 0])
    for idx in range(x.shape[1]):
        state = a[:, idx].unsqueeze(-1) * state + x[:, idx]
        y.append(state)
    out = torch.stack(y, dim=1)
    return out.squeeze(-1) if squeeze else out


def serial_mat_scan(a, x):
    squeeze = x.ndim == 4
    if squeeze:
        x = x.unsqueeze(-1)
    y = []
    state = torch.zeros_like(x[:, 0])
    for idx in range(x.shape[1]):
        state = torch.matmul(a[:, idx], state) + x[:, idx]
        y.append(state)
    out = torch.stack(y, dim=1)
    return out.squeeze(-1) if squeeze else out


def manual_forward(model, x, dt):
    batch_size, steps = x.shape[:2]
    dt_seq = model._normalize_dt(dt, batch_size, steps, x.device)
    model._validate_dt(dt_seq)
    latent = model.encoder(x)
    z_skip = latent
    states = model._init_states(batch_size, x.device, latent.dtype)
    for block_idx, block in enumerate(model.blocks):
        latent, h_next = block(latent, states[block_idx], dt_seq)
        states[block_idx] = h_next
    latent = model._apply_decoder_skip(latent, z_skip)
    out = model.decoder(latent)
    mask = dt_seq.abs() <= 1e-12
    while mask.ndim < out.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, x, out)
