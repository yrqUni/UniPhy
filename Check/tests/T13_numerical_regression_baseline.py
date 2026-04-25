import argparse
import hashlib
import sys
from importlib import resources
from pathlib import Path

import torch

from Check.utils import REPO_DIR, build_check_model, write_result

TEST_ID = "T13"
GOLDEN_DIR = REPO_DIR / "Check" / "golden"
GOLDEN_PATH = GOLDEN_DIR / "golden.pt"
PACKAGED_GOLDEN = "golden.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_outputs(device):
    model = build_check_model(device, torch.float64)
    model.eval()
    torch.manual_seed(0)
    batch_size = 2
    x_pixels = torch.randn(batch_size, 3, 4, 32, 32, device=device, dtype=torch.float64)
    dt = torch.tensor(
        [[1.0, 0.75, 1.25], [1.0, 1.5, 0.5]],
        device=device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        out_fwd = model.forward(x_pixels, dt)
        out_fwd_r = out_fwd.real if out_fwd.is_complex() else out_fwd
        dt_list = [
            torch.tensor([0.5, 1.0], device=device, dtype=torch.float64),
            torch.tensor([1.5, 0.5], device=device, dtype=torch.float64),
        ]
        out_roll = model.forward_rollout(x_pixels[:, :2], dt[:, :2], dt_list)
        out_roll_r = out_roll.real if out_roll.is_complex() else out_roll
    return {"fwd": out_fwd_r.cpu(), "roll": out_roll_r.cpu()}


def resolve_golden_path():
    if GOLDEN_PATH.exists():
        return GOLDEN_PATH
    try:
        package_path = resources.files("Check.golden") / PACKAGED_GOLDEN
    except ModuleNotFoundError:
        return None
    return Path(str(package_path)) if package_path.is_file() else None


def run(regenerate=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if regenerate:
        payload = compute_outputs(device)
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(payload, GOLDEN_PATH)
        sha = file_sha256(GOLDEN_PATH)
        detail = f"golden regenerated sha256={sha}"
        return "PASS", 0.0, detail
    golden_path = resolve_golden_path()
    if golden_path is None:
        detail = f"missing golden file: {GOLDEN_PATH}"
        return "FAIL", "-", detail
    payload = compute_outputs(device)
    golden = torch.load(golden_path, map_location="cpu", weights_only=True)
    err_fwd = float((payload["fwd"] - golden["fwd"]).abs().max().item())
    err_roll = float((payload["roll"] - golden["roll"]).abs().max().item())
    max_err = max(err_fwd, err_roll)
    status = "PASS" if max_err < 1e-8 else "FAIL"
    detail = f"err_fwd={err_fwd:.2e} err_roll={err_roll:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    args = parse_args()
    status, max_error, detail = run(regenerate=args.regenerate)
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
