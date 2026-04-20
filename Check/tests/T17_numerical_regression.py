import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, make_tiny_model, write_result
else:
    from ..utils import REPO_DIR, make_tiny_model, write_result

import torch

TEST_ID = "T17"
GOLDEN_PATH = REPO_DIR / "Check" / "golden" / "golden.pt"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_tiny_model(device, seed=42)
    model.eval()
    torch.manual_seed(0)
    batch_size, steps = 2, 2
    x_pixels = torch.randn(batch_size, steps, 4, 721, 1440, device=device)
    dt = torch.full((batch_size, steps), 6.0, device=device)
    with torch.no_grad():
        out_fwd = model.forward(x_pixels, dt)
        out_fwd_r = out_fwd.real if out_fwd.is_complex() else out_fwd
        dt_list = [torch.full((batch_size,), 6.0, device=device)] * 2
        out_roll = model.forward_rollout(x_pixels[:, :1], dt[:, :1], dt_list)
        out_roll_r = out_roll.real if out_roll.is_complex() else out_roll
    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"fwd": out_fwd_r.cpu(), "roll": out_roll_r.cpu()}, GOLDEN_PATH)
        return "PASS", 0.0, "golden values saved"
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
    err_fwd = float((out_fwd_r.cpu() - golden["fwd"]).abs().max().item())
    err_roll = float((out_roll_r.cpu() - golden["roll"]).abs().max().item())
    max_err = max(err_fwd, err_roll)
    status = "PASS" if max_err < 1e-5 else "FAIL"
    detail = f"err_fwd={err_fwd:.2e} err_roll={err_roll:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
