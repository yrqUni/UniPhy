import sys

import torch
import torch.nn.functional as F

from Model.UniPhy.UniPhyOps import TemporalPropagator
from dt_check.utils import fit_log_slope, inverse_softplus, write_result

TEST_ID = "T02_ssm_discretisation"


LAMBDAS = [complex(-0.5, 0.0), complex(-0.1, 0.3), complex(-2.0, 1.0)]
DT_RATIOS = [0.01, 0.1, 0.5, 1.0, 4.0]


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    prop = TemporalPropagator(dim=16, dt_ref=1.0, init_noise_scale=1e-4).to(device)
    prop.eval()
    records = []
    slopes = []
    max_err = 0.0
    with torch.no_grad():
        for lam in LAMBDAS:
            target_re = torch.full((16,), -lam.real, device=device, dtype=torch.float64)
            prop.lam_re.copy_(inverse_softplus(target_re).to(torch.float32))
            prop.lam_im.copy_(torch.full((16,), lam.imag, device=device))
            lam_t = torch.full((16,), lam, device=device, dtype=torch.complex64)
            lam_errors = []
            for dt_ratio in DT_RATIOS:
                h0 = torch.randn(16, device=device, dtype=torch.float32)
                h0 = torch.complex(h0, torch.randn_like(h0))
                u0 = torch.randn(16, device=device, dtype=torch.float32)
                u0 = torch.complex(u0, torch.randn_like(u0))
                dt_tensor = torch.tensor([dt_ratio], device=device)
                decay, forcing = prop._compute_exp_operators(dt_tensor)
                code_result = decay[0] * h0 + forcing[0] * u0
                exp_arg = lam_t * dt_ratio
                analytic = torch.exp(exp_arg) * h0 + torch.where(
                    exp_arg == 0,
                    torch.ones_like(exp_arg),
                    torch.expm1(exp_arg) / exp_arg,
                ) * dt_ratio * u0
                err = float((analytic - code_result).abs().max().item())
                records.append((lam, dt_ratio, err))
                lam_errors.append(max(err, 1e-30))
                max_err = max(max_err, err)
            slopes.append(fit_log_slope(DT_RATIOS, lam_errors))
    slope_ok = all(slope >= 1.8 for slope in slopes)
    status = "PASS" if max_err < 1e-5 and slope_ok else "FAIL"
    slope_text = ",".join(f"{slope:.2f}" for slope in slopes)
    detail = f"max_err={max_err:.2e} slopes={slope_text}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
