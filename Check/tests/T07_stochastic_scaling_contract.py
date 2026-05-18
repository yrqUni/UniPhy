import sys
import math
import cmath

import torch

from Check.utils import write_result
from Model.UniPhy.UniPhyOps import (
    TemporalPropagator,
    _cholesky_components,
    _compute_sde_covariance,
)

TEST_ID = "T07"


def _isotropic_std(lam_phys, dt_phys, base_noise):
    arg = 2.0 * lam_phys * dt_phys
    if abs(arg) < 1e-7:
        variance = dt_phys
    else:
        variance = math.expm1(arg) / (2.0 * lam_phys)
    return base_noise * math.sqrt(max(variance, 0.0))


def _diagonal_std(lam_phys_val, dt_phys_val, base_val, device):
    lam_phys = torch.complex(
        torch.tensor([lam_phys_val], dtype=torch.float64, device=device),
        torch.tensor([0.0], dtype=torch.float64, device=device),
    )
    dt_phys = torch.tensor([dt_phys_val], dtype=torch.float64, device=device)
    base = torch.tensor([base_val], dtype=torch.float64, device=device)
    var_re, _, _ = _compute_sde_covariance(lam_phys, dt_phys, base, base)
    return float(torch.sqrt(var_re).item())


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt_small = [1e-6, 1e-5, 1e-4, 1e-3]
    scales_a = [_diagonal_std(-1.0, dt, 1.0, device) for dt in dt_small]

    lam_vals = [-0.1, -1.0, -10.0, -100.0]
    scales_b = [_diagonal_std(lam, 1.0, 1.0, device) for lam in lam_vals]

    scale_c = _diagonal_std(0.0, 1.0, 1.0, device)
    err_c = abs(scale_c - 1.0)

    lam_phys_test = -0.5 / 6.0
    theory_err = 0.0
    for dt_hours in [1.0, 6.0, 12.0, 24.0, 48.0]:
        code_scale = _diagonal_std(lam_phys_test, dt_hours, 1.0, device)
        theory_scale = _isotropic_std(lam_phys_test, dt_hours, 1.0)
        theory_err = max(theory_err, abs(code_scale - theory_scale))

    prop = TemporalPropagator(dim=8, dt_ref=1.0, init_noise_scale=0.1).to(device)
    noise_real = torch.randn(2, 3, 4, 5, 8, device=device)
    noise_complex = torch.complex(noise_real, torch.randn_like(noise_real))
    real_rejected = False
    try:
        prop._normalize_explicit_noise(noise_real, torch.complex64)
    except TypeError:
        real_rejected = True
    norm_complex = prop._normalize_explicit_noise(noise_complex, torch.complex64)
    complex_imag = float(norm_complex.imag.abs().mean().item())

    seq_shape = (2, 3, 4, 5, 8)
    dt_seq = torch.full((2, 3), 0.5, device=device)
    seq_noise = torch.complex(
        torch.randn(seq_shape, device=device),
        torch.randn(seq_shape, device=device),
    )
    stochastic_seq = prop.generate_stochastic_term(
        seq_shape,
        dt_seq,
        torch.complex64,
        noise=seq_noise,
    )
    seq_zero = prop.generate_stochastic_term(
        seq_shape,
        dt_seq,
        torch.complex64,
        noise=torch.zeros(seq_shape, device=device, dtype=torch.complex64),
    )
    seq_none = prop.generate_stochastic_term(
        seq_shape,
        dt_seq,
        torch.complex64,
        noise=None,
    )

    step_shape = (2, 4, 5, 8)
    dt_step = torch.full((2,), 0.5, device=device)
    step_noise = torch.complex(
        torch.randn(step_shape, device=device),
        torch.randn(step_shape, device=device),
    )
    stochastic_step = prop.generate_stochastic_term(
        step_shape,
        dt_step,
        torch.complex64,
        noise=step_noise,
    )
    step_zero = prop.generate_stochastic_term(
        step_shape,
        dt_step,
        torch.complex64,
        noise=torch.zeros(step_shape, device=device, dtype=torch.complex64),
    )
    step_none = prop.generate_stochastic_term(
        step_shape,
        dt_step,
        torch.complex64,
        noise=None,
    )

    stochastic_step_same = prop.generate_stochastic_term(
        step_shape,
        dt_seq[:, 0],
        torch.complex64,
        noise=seq_noise[:, 0],
    )
    consistency_err = float(
        (stochastic_seq[:, 0] - stochastic_step_same).abs().max().item()
    )

    seq_shape_ok = stochastic_seq.shape == torch.Size(seq_shape)
    step_shape_ok = stochastic_step.shape == torch.Size(step_shape)

    mc_dtype = torch.float64
    lam_mc = torch.complex(
        torch.tensor([-0.5], dtype=mc_dtype, device=device),
        torch.tensor([1.5], dtype=mc_dtype, device=device),
    )
    sigma_re_param = torch.tensor([1.0], dtype=mc_dtype, device=device)
    sigma_im_param = torch.tensor([0.3], dtype=mc_dtype, device=device)
    dt_phys_mc = torch.tensor([1.0], dtype=mc_dtype, device=device)

    var_re_th, var_im_th, cov_th = _compute_sde_covariance(
        lam_mc,
        dt_phys_mc,
        sigma_re_param,
        sigma_im_param,
    )
    sigma_re_chol, cross_factor_chol, sigma_im_chol = _cholesky_components(
        var_re_th,
        var_im_th,
        cov_th,
    )

    n_samples = 200_000
    torch.manual_seed(20260511)
    xi_re = torch.randn(n_samples, 1, dtype=mc_dtype, device=device)
    xi_im = torch.randn(n_samples, 1, dtype=mc_dtype, device=device)
    eps_re_samples = sigma_re_chol * xi_re
    eps_im_samples = cross_factor_chol * xi_re + sigma_im_chol * xi_im

    var_re_emp = eps_re_samples.var(dim=0, unbiased=False)
    var_im_emp = eps_im_samples.var(dim=0, unbiased=False)
    cov_emp = (eps_re_samples * eps_im_samples).mean(dim=0)

    mc_err_var_re = float((var_re_emp - var_re_th).abs().max().item())
    mc_err_var_im = float((var_im_emp - var_im_th).abs().max().item())
    mc_err_cov = float((cov_emp - cov_th).abs().max().item())
    mc_max_err = max(mc_err_var_re, mc_err_var_im, mc_err_cov)

    lam_py = complex(-0.5, 1.5)
    dt_py = 1.0
    sr, si = 1.0, 0.3
    R_py = (math.exp(2 * lam_py.real * dt_py) - 1) / (2 * lam_py.real)
    P_py = (cmath.exp(2 * lam_py * dt_py) - 1) / (2 * lam_py)
    var_re_ref = (
        0.5 * (sr * sr + si * si) * R_py + 0.5 * (sr * sr - si * si) * P_py.real
    )
    var_im_ref = (
        0.5 * (sr * sr + si * si) * R_py - 0.5 * (sr * sr - si * si) * P_py.real
    )
    cov_ref = 0.5 * (sr * sr - si * si) * P_py.imag
    mc_th_err = max(
        abs(float(var_re_th.item()) - var_re_ref),
        abs(float(var_im_th.item()) - var_im_ref),
        abs(float(cov_th.item()) - cov_ref),
    )

    psd_residual = float((var_re_th * var_im_th - cov_th * cov_th).min().item())
    mc_tol = (
        6.0
        * max(float(var_re_th.item()), float(var_im_th.item()))
        * (2.0 / n_samples) ** 0.5
    )
    cross_factor_min = float(cross_factor_chol.abs().min().item())

    passed = (
        all(scales_a[i] < scales_a[i + 1] for i in range(3))
        and all(scales_b[i] > scales_b[i + 1] for i in range(3))
        and err_c < 1e-5
        and theory_err < 1e-5
        and real_rejected
        and complex_imag > 1e-3
        and float(seq_zero.abs().max().item()) == 0.0
        and float(step_zero.abs().max().item()) == 0.0
        and float(seq_none.abs().max().item()) == 0.0
        and float(step_none.abs().max().item()) == 0.0
        and float(stochastic_seq.abs().mean().item()) > 1e-3
        and float(stochastic_step.abs().mean().item()) > 1e-3
        and seq_shape_ok
        and step_shape_ok
        and consistency_err < 1e-6
        and mc_max_err < mc_tol
        and mc_th_err < 1e-6
        and psd_residual >= -1e-12
        and cross_factor_min > 1e-4
    )
    max_err = max(err_c, theory_err, consistency_err, mc_max_err, mc_th_err)
    detail = (
        f"err_c={err_c:.2e} theory_err={theory_err:.2e} "
        f"real_rejected={real_rejected} complex_imag={complex_imag:.2e} "
        f"seq_mean={float(stochastic_seq.abs().mean().item()):.2e} "
        f"step_mean={float(stochastic_step.abs().mean().item()):.2e} "
        f"consistency_err={consistency_err:.2e} "
        f"seq_shape_ok={seq_shape_ok} step_shape_ok={step_shape_ok} "
        f"mc_max_err={mc_max_err:.2e} mc_tol={mc_tol:.2e} "
        f"mc_th_err={mc_th_err:.2e} psd_residual={psd_residual:.2e} "
        f"cross_factor_min={cross_factor_min:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
