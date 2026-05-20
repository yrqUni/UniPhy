import sys

import torch

from Check.utils import inverse_softplus, write_result
from Model.UniPhy.UniPhyOps import (
    TemporalPropagator,
    _compute_sde_covariance,
    _etd2_coefficients,
)

TEST_ID = "T15"


def _build_with_lam_phys(lam_phys, dim, dt_ref, device):
    prop = (
        TemporalPropagator(dim=dim, dt_ref=dt_ref, init_noise_scale=1e-4)
        .to(device)
        .double()
    )
    with torch.no_grad():
        target_re = torch.full(
            (dim,),
            -lam_phys.real * dt_ref,
            device=device,
            dtype=torch.float64,
        )
        prop.lam_re_raw.copy_(inverse_softplus(target_re))
        prop.lam_im_raw.copy_(
            torch.full(
                (dim,),
                lam_phys.imag * dt_ref,
                device=device,
                dtype=torch.float64,
            ),
        )
    return prop


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 8
    lam_phys = complex(-0.08, 0.05)
    dt_phys_values = [1.0, 6.0, 12.0, 24.0]
    dt_ref_values = [1.0, 3.0, 6.0, 24.0]

    strict_err = 0.0
    convention_err = 0.0
    with torch.no_grad():
        lam_t = torch.complex(
            torch.tensor([lam_phys.real], dtype=torch.float64, device=device),
            torch.tensor([lam_phys.imag], dtype=torch.float64, device=device),
        )
        for dt_phys_val in dt_phys_values:
            dt_t = torch.tensor([dt_phys_val], device=device, dtype=torch.float64)

            decay_strict, alpha_strict, beta_strict = _etd2_coefficients(lam_t, dt_t)
            base_re = torch.tensor([[1.0]], dtype=torch.float64, device=device)
            var_re_strict, var_im_strict, cov_strict = _compute_sde_covariance(
                lam_t.unsqueeze(0),
                dt_t.unsqueeze(-1),
                base_re,
                base_re,
            )

            for dt_ref in dt_ref_values:
                prop = _build_with_lam_phys(lam_phys, dim, dt_ref, device)

                lam_recovered = prop._get_effective_lambda()
                decay_low, alpha_low, beta_low = _etd2_coefficients(
                    lam_recovered[:1],
                    dt_t,
                )
                strict_err = max(
                    strict_err,
                    float((decay_low - decay_strict).abs().max().item()),
                    float((alpha_low - alpha_strict).abs().max().item()),
                    float((beta_low - beta_strict).abs().max().item()),
                )

                decay_p, alpha_p, beta_p = prop.get_etd2_operators(dt_t)
                expect_alpha = alpha_strict / dt_ref
                expect_beta = beta_strict / dt_ref
                convention_err = max(
                    convention_err,
                    float((decay_p[0] - decay_strict).abs().max().item()),
                    float((alpha_p[0] - expect_alpha).abs().max().item()),
                    float((beta_p[0] - expect_beta).abs().max().item()),
                )

                sigma_re_p, _, _ = prop._compute_noise_scales(dt_t.unsqueeze(-1))

                base = prop.base_noise_re.abs()[0].item()
                expect_var_re = (base * base / dt_ref) * var_re_strict[0, 0]
                expect_sigma_re = float(expect_var_re.sqrt().item())
                actual_sigma_re = float(sigma_re_p[0, 0].item())
                convention_err = max(
                    convention_err,
                    abs(actual_sigma_re - expect_sigma_re),
                )

    max_err = max(strict_err, convention_err)
    passed = strict_err < 1e-12 and convention_err < 1e-10
    detail = f"strict_err={strict_err:.2e} convention_err={convention_err:.2e}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
