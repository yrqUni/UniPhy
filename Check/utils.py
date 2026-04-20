import os
import subprocess
from pathlib import Path

import torch

from Model.UniPhy.ModelUniPhy import UniPhyModel

LOG_DIR = "/nfs/A/logs"
REPO_DIR = Path(__file__).resolve().parents[1]


def format_max_error(value):
    if value in {None, "-"}:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.6e}"


def write_result(test_id, status, max_error, detail, log_dir=LOG_DIR):
    if status not in {"PASS", "FAIL", "SKIP"}:
        raise ValueError(f"invalid status: {status}")
    os.makedirs(log_dir, exist_ok=True)
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


def assert_close(a, b, atol, label=""):
    del label
    diff = (a - b).abs()
    if diff.numel() == 0:
        return 0.0, True
    max_diff = float(diff.max().item())
    passed = bool(torch.isfinite(diff).all().item() and max_diff <= atol)
    return max_diff, passed


def sequential_ssm_recurrence(decay_seq, x_seq, h0):
    rows = []
    state = h0
    for step in range(decay_seq.shape[1]):
        state = decay_seq[:, step] * state + x_seq[:, step]
        rows.append(state)
    return torch.stack(rows, dim=1)


def make_tiny_model(device, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = UniPhyModel(
        in_channels=4,
        out_channels=4,
        embed_dim=8,
        expand=2,
        depth=1,
        patch_size=[7, 15],
        img_height=721,
        img_width=1440,
        dt_ref=6.0,
        init_noise_scale=1e-4,
    )
    return model.to(device).eval()


def select_free_gpu(min_free_mb=2000):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        idx_str, free_str = [part.strip() for part in line.split(",", maxsplit=1)]
        rows.append((int(idx_str), int(free_str)))
    if not rows:
        raise RuntimeError("no gpu detected")
    gpu_idx, free_mb = max(rows, key=lambda item: item[1])
    if free_mb < min_free_mb:
        raise RuntimeError(f"no gpu has >= {min_free_mb} MB free")
    return gpu_idx


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
