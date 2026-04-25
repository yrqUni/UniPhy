import sys

import torch

from Check.utils import inverse_softplus, write_result
from Model.UniPhy.UniPhyOps import TemporalPropagator, _safe_forcing

TEST_ID = "T01"
LAMBDAS = [complex(-0.5, 0.0), complex(-0.1, 0.3), complex(-2.0, 1.0)]
DT_RATIOS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    prop = TemporalPropagator(dim=16, dt_ref=1.0, init_noise_scale=1e-4).to(device)
    prop.eval()
    z_real = torch.logspace(-10, -4, 200, dtype=torch.float64, device=device)
    z = torch.complex(z_real, torch.zeros_like(z_real))
    phi1_ref = torch.expm1(z) / z
    phi1_ref = torch.where(z == 0, torch.ones_like(phi1_ref), phi1_ref)
    phi1_err = float(
        (_safe_forcing(z, torch.ones_like(z_real)) - phi1_ref).abs().max().item()
    )

    max_err = phi1_err
    max_dt_order_err = 0.0
    with torch.no_grad():
        for lam in LAMBDAS:
            target_re = torch.full(
                (16,),
                -lam.real,
                device=device,
                dtype=torch.float64,
            )
            prop.lam_re.copy_(inverse_softplus(target_re).to(torch.float32))
            prop.lam_im.copy_(torch.full((16,), lam.imag, device=device))
            lam_t = torch.full((16,), lam, device=device, dtype=torch.complex128)
            h0 = torch.complex(
                torch.randn(16, device=device, dtype=torch.float64),
                torch.randn(16, device=device, dtype=torch.float64),
            )
            u0 = torch.complex(
                torch.randn(16, device=device, dtype=torch.float64),
                torch.randn(16, device=device, dtype=torch.float64),
            )
            for dt_ratio in DT_RATIOS:
                dt_tensor = torch.tensor([dt_ratio], device=device, dtype=torch.float64)
                decay, forcing = prop._compute_exp_operators(dt_tensor)
                code_result = decay[0].to(torch.complex128) * h0 + forcing[0] * u0
                exp_arg = lam_t * dt_ratio
                analytic = (
                    torch.exp(exp_arg) * h0
                    + torch.where(
                        exp_arg == 0,
                        torch.ones_like(exp_arg),
                        torch.expm1(exp_arg) / exp_arg,
                    )
                    * dt_ratio
                    * u0
                )
                err = float((analytic - code_result).abs().max().item())
                max_err = max(max_err, err)
                linearized = h0 + dt_ratio * (lam_t * h0 + u0)
                dt_order_err = float((code_result - linearized).abs().max().item())
                max_dt_order_err = max(max_dt_order_err, dt_order_err)

    dt_zero = torch.tensor([1e-8, 1e-7, 1e-6], device=device, dtype=torch.float64)
    lam_zero = torch.complex(
        torch.full((3,), -0.5, device=device, dtype=torch.float64),
        torch.full((3,), 0.3, device=device, dtype=torch.float64),
    )
    decay_zero = torch.exp(lam_zero * dt_zero)
    forcing_zero = _safe_forcing(lam_zero * dt_zero, dt_zero)
    zero_identity_err = float((decay_zero - 1).abs().max().item())
    zero_forcing_err = float((forcing_zero - dt_zero).abs().max().item())
    max_err = max(max_err, zero_identity_err, zero_forcing_err)

    passed = max_err < 1e-5
    detail = (
        f"max_err={max_err:.2e} phi1_err={phi1_err:.2e} "
        f"zero_identity_err={zero_identity_err:.2e} "
        f"zero_forcing_err={zero_forcing_err:.2e} "
        f"max_dt_order_err={max_dt_order_err:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
