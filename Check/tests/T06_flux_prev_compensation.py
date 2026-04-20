import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import complex_randn, write_result
else:
    from ..utils import complex_randn, write_result

import torch
from Model.UniPhy.PScan import pscan

TEST_ID = "T06"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size, steps, state_dim = 2, 12, 8
    decay_seq = 0.2 * complex_randn((batch_size, steps, state_dim, 1), device)
    x_scan = complex_randn((batch_size, steps, state_dim, 1), device)
    flux_prev = complex_randn((batch_size, state_dim), device)
    flux_pscan = pscan(decay_seq, x_scan).squeeze(-1)
    decay_sq = decay_seq.squeeze(-1)
    decay_cum = torch.cumprod(decay_sq, dim=1)
    flux_a = flux_pscan + flux_prev.unsqueeze(1) * decay_cum
    flux = flux_prev.clone()
    rows = []
    for t in range(steps):
        flux = decay_sq[:, t] * flux + x_scan[:, t, :, 0]
        rows.append(flux.clone())
    flux_b = torch.stack(rows, dim=1)
    err = float((flux_a - flux_b).abs().max().item())
    status = "PASS" if err < 1e-4 else "FAIL"
    detail = f"max_err={err:.2e}"
    return status, err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
