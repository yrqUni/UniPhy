import sys

import torch

from Model.UniPhy.UniPhyOps import GlobalFluxTracker
from dt_check.utils import write_result

TEST_ID = "T07_cumprod_decay_purity"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    tracker = GlobalFluxTracker(dim=16, dt_ref=6.0).to(device)
    tracker.eval()
    batch_size, steps = 2, 8
    x_mean_seq = torch.randn(batch_size, steps, 16, device=device)
    x_mean_seq = torch.complex(x_mean_seq, torch.randn_like(x_mean_seq))
    dt_seq = torch.rand(batch_size, steps, device=device) * 12 + 1
    with torch.no_grad():
        a_flux, x_flux = tracker.get_scan_operators(x_mean_seq, dt_seq)
        dt_ratio = dt_seq.unsqueeze(-1) / tracker.dt_ref
        lam = tracker._get_continuous_params()
        expected_decay = torch.exp(lam * dt_ratio).unsqueeze(-1)
    err = float((a_flux - expected_decay).abs().max().item())
    purity_ok = bool(((x_flux - a_flux).abs() > 1e-8).any().item())
    status = "PASS" if err < 1e-6 and purity_ok else "FAIL"
    detail = f"max_err={err:.2e} purity_ok={purity_ok}"
    return status, err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
