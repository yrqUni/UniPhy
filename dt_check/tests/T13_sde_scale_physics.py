import sys

import torch

from Model.UniPhy.UniPhyOps import _compute_sde_scale
from dt_check.utils import write_result

TEST_ID = "T13_sde_scale_physics"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_noise = torch.ones(8, device=device)
    dt_small = [1e-6, 1e-5, 1e-4, 1e-3]
    scales_a = [
        float(
            _compute_sde_scale(
                torch.full((8,), -1.0, device=device),
                torch.tensor(dt, device=device),
                base_noise,
            ).max().item()
        )
        for dt in dt_small
    ]
    check_a = all(scale < dt * 2 for scale, dt in zip(scales_a, dt_small))
    lam_vals = [-0.1, -1.0, -10.0, -100.0]
    scales_b = [
        float(
            _compute_sde_scale(
                torch.full((8,), lam, device=device),
                torch.tensor(1.0, device=device),
                base_noise,
            ).max().item()
        )
        for lam in lam_vals
    ]
    check_b = all(scales_b[i] > scales_b[i + 1] for i in range(3))
    scale_c = _compute_sde_scale(
        torch.zeros(8, device=device), torch.tensor(1.0, device=device), base_noise
    )
    analytic = base_noise * torch.sqrt(torch.tensor(1.0, device=device))
    err_c = float((scale_c - analytic).abs().max().item())
    lam_re = torch.full((8,), -0.5, device=device)
    dt_vals = [0.5, 1.0, 2.0, 4.0]
    scales_d = [
        float(
            _compute_sde_scale(
                lam_re,
                torch.tensor(dt, device=device),
                base_noise,
            ).max().item()
        )
        for dt in dt_vals
    ]
    check_d = all(scales_d[i] < scales_d[i + 1] for i in range(3))
    passed = check_a and check_b and err_c < 1e-5 and check_d
    status = "PASS" if passed else "FAIL"
    detail = f"err_c={err_c:.2e}"
    return status, err_c, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
