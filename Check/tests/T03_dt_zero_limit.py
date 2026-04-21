import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import write_result
else:
    from ..utils import write_result

import torch
from Model.UniPhy.UniPhyOps import _safe_forcing

TEST_ID = "T03"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dt_values = [1e-8, 1e-7, 1e-6, 1e-5]
    lam = torch.complex(
        torch.full((8,), -0.5, device=device, dtype=torch.float64),
        torch.full((8,), 0.3, device=device, dtype=torch.float64),
    )
    max_residual = 0.0
    passed = True
    for dt_ratio in dt_values:
        dt_t = torch.tensor([dt_ratio], device=device, dtype=torch.float64)
        exp_arg = lam * dt_ratio
        decay = torch.exp(exp_arg)
        forcing = _safe_forcing(exp_arg, dt_t)
        decay_ok = bool(((decay - 1).abs() < 1e-4).all().item())
        forcing_ok = bool((forcing.abs() < dt_ratio * 2).all().item())
        h0 = torch.randn(8, device=device, dtype=torch.float64)
        h0 = torch.complex(h0, torch.randn_like(h0))
        u = torch.randn(8, device=device, dtype=torch.float64)
        u = torch.complex(u, torch.randn_like(u))
        h_next = decay * h0 + forcing * u
        h_linear = h0 + dt_ratio * (lam * h0 + u)
        residual = float((h_next - h_linear).abs().max().item())
        max_residual = max(max_residual, residual)
        passed = passed and decay_ok and forcing_ok and residual < dt_ratio**2 * 100
    status = "PASS" if passed else "FAIL"
    detail = f"max_residual={max_residual:.2e}"
    return status, max_residual, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
