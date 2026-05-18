import sys

import torch

from Check.utils import inverse_softplus, write_result
from Model.UniPhy.UniPhyOps import (
    TemporalPropagator,
    _etd2_coefficients,
    _safe_phi1,
    _safe_phi2,
)

TEST_ID = "T01"


LAMBDA_PHYS = [complex(-0.5, 0.0), complex(-0.1, 0.3), complex(-2.0, 1.0)]

DT_PHYS_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
DT_REF = 1.0


def _phi_validity_errors(device):
    z_real = torch.logspace(-10, -4, 200, dtype=torch.float64, device=device)
    z = torch.complex(z_real, torch.zeros_like(z_real))
    dt_phys = torch.ones_like(z_real)
    phi1_ref = torch.expm1(z) / z
    phi1_ref = torch.where(z == 0, torch.ones_like(phi1_ref), phi1_ref)
    phi1_err = float((_safe_phi1(z, dt_phys) - phi1_ref).abs().max().item())
    phi2_ref = (torch.expm1(z) - z) / (z * z)
    phi2_ref = torch.where(
        z.abs() < 1e-12,
        torch.full_like(phi2_ref, 0.5),
        phi2_ref,
    )
    phi2_err = float((_safe_phi2(z, dt_phys) - phi2_ref).abs().max().item())
    return phi1_err, phi2_err


def _set_lambda(prop, lam_phys, dim, device):
    target_re = torch.full(
        (dim,),
        -lam_phys.real * prop.dt_ref,
        device=device,
        dtype=torch.float64,
    )
    prop.lam_re_raw.copy_(inverse_softplus(target_re).to(torch.float32))
    prop.lam_im_raw.copy_(
        torch.full(
            (dim,),
            lam_phys.imag * prop.dt_ref,
            device=device,
            dtype=prop.lam_im_raw.dtype,
        ),
    )


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    dim = 16
    prop = TemporalPropagator(dim=dim, dt_ref=DT_REF, init_noise_scale=1e-4).to(device)
    prop.eval()

    phi1_err, phi2_err = _phi_validity_errors(device)

    etd2_linear_err = 0.0
    etd2_constant_check_err = 0.0
    with torch.no_grad():
        for lam in LAMBDA_PHYS:
            _set_lambda(prop, lam, dim, device)
            lam_check = prop._get_effective_lambda()
            assert torch.allclose(
                lam_check.real,
                torch.full_like(lam_check.real, lam.real),
                atol=1e-5,
            ), f"lam_re mismatch: got {lam_check.real[0].item()} want {lam.real}"
            lam_t = torch.full((dim,), lam, device=device, dtype=torch.complex128)
            h0 = torch.complex(
                torch.randn(dim, device=device, dtype=torch.float64),
                torch.randn(dim, device=device, dtype=torch.float64),
            )
            u0 = torch.complex(
                torch.randn(dim, device=device, dtype=torch.float64),
                torch.randn(dim, device=device, dtype=torch.float64),
            )
            u1 = torch.complex(
                torch.randn(dim, device=device, dtype=torch.float64),
                torch.randn(dim, device=device, dtype=torch.float64),
            )
            for dt_phys_val in DT_PHYS_VALUES:
                dt_tensor = torch.tensor(
                    [dt_phys_val],
                    device=device,
                    dtype=torch.float64,
                )
                exp_arg = lam_t * dt_phys_val
                exp_decay = torch.exp(exp_arg)
                phi1_times_dt = torch.where(
                    exp_arg == 0,
                    torch.full_like(exp_arg, dt_phys_val),
                    torch.expm1(exp_arg) / exp_arg * dt_phys_val,
                )
                phi2_times_dt = torch.where(
                    exp_arg.abs() < 1e-12,
                    torch.full_like(exp_arg, dt_phys_val * 0.5),
                    (torch.expm1(exp_arg) - exp_arg)
                    / (exp_arg * exp_arg)
                    * dt_phys_val,
                )

                decay, alpha, beta = prop.get_etd2_operators(dt_tensor)
                decay = decay[0].to(torch.complex128)
                alpha = alpha[0].to(torch.complex128)
                beta = beta[0].to(torch.complex128)

                analytic_linear = (
                    exp_decay * h0
                    + (phi1_times_dt - phi2_times_dt) * u0
                    + phi2_times_dt * u1
                )
                h_etd2 = decay * h0 + alpha * u0 + beta * u1
                etd2_linear_err = max(
                    etd2_linear_err,
                    float((h_etd2 - analytic_linear).abs().max().item()),
                )

                analytic_constant = exp_decay * h0 + phi1_times_dt * u0
                h_etd2_constant = decay * h0 + (alpha + beta) * u0
                etd2_constant_check_err = max(
                    etd2_constant_check_err,
                    float((h_etd2_constant - analytic_constant).abs().max().item()),
                )

    dt_zero = torch.tensor([1e-8, 1e-7, 1e-6], device=device, dtype=torch.float64)
    lam_zero = torch.complex(
        torch.full((3,), -0.5, device=device, dtype=torch.float64),
        torch.full((3,), 0.3, device=device, dtype=torch.float64),
    )
    decay_zero = torch.exp(lam_zero * dt_zero)
    zero_identity_err = float((decay_zero - 1).abs().max().item())

    _, alpha_zero, beta_zero = _etd2_coefficients(lam_zero, dt_zero)
    limit_sum_err = float((alpha_zero + beta_zero - dt_zero).abs().max().item())

    max_err = max(
        phi1_err,
        phi2_err,
        etd2_linear_err,
        etd2_constant_check_err,
        zero_identity_err,
        limit_sum_err,
    )
    passed = max_err < 1e-5
    detail = (
        f"phi1_err={phi1_err:.2e} phi2_err={phi2_err:.2e} "
        f"etd2_linear_err={etd2_linear_err:.2e} "
        f"etd2_constant_err={etd2_constant_check_err:.2e} "
        f"zero_identity_err={zero_identity_err:.2e} "
        f"limit_sum_err={limit_sum_err:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
