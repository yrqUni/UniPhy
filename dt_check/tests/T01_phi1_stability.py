import sys

import torch

from Model.UniPhy.UniPhyOps import _safe_forcing
from dt_check.utils import write_result

TEST_ID = "T01_phi1_stability"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    z_real = torch.logspace(-10, -4, 200, dtype=torch.float64, device=device)
    z_f64 = z_real.to(torch.complex128)
    z_f32 = z_real.float().to(torch.complex64)
    dt64 = torch.ones_like(z_f64, dtype=torch.float64, device=device)
    dt32 = torch.ones_like(z_real.float(), device=device)
    ref = torch.expm1(z_f64) / z_f64
    got = _safe_forcing(z_f32, dt32)
    rel_err = (got.abs().to(torch.float64) - ref.abs()).abs() / (
        ref.abs() + 1e-30
    )
    max_rel_err = float(rel_err.max().item())
    boundary = torch.tensor([1e-7, 0.999999e-7, 1.000001e-7], device=device)
    boundary_complex = boundary.to(torch.complex64)
    boundary_vals = _safe_forcing(boundary_complex, torch.ones_like(boundary))
    boundary_ok = bool(torch.isfinite(boundary_vals.real).all().item())
    status = "PASS" if max_rel_err < 1e-5 and boundary_ok else "FAIL"
    detail = f"max_rel_err={max_rel_err:.2e} boundary_ok={boundary_ok}"
    return status, max_rel_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
