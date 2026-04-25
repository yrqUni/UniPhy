import sys

import torch

from Check.utils import write_result
from Model.UniPhy.UniPhyOps import TemporalPropagator, _compute_sde_scale

TEST_ID = "T07"


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
            )
            .max()
            .item()
        )
        for dt in dt_small
    ]
    lam_vals = [-0.1, -1.0, -10.0, -100.0]
    scales_b = [
        float(
            _compute_sde_scale(
                torch.full((8,), lam, device=device),
                torch.tensor(1.0, device=device),
                base_noise,
            )
            .max()
            .item()
        )
        for lam in lam_vals
    ]
    scale_c = _compute_sde_scale(
        torch.zeros(8, device=device),
        torch.tensor(1.0, device=device),
        base_noise,
    )
    err_c = float((scale_c - base_noise).abs().max().item())

    prop = TemporalPropagator(dim=8, dt_ref=1.0, init_noise_scale=0.1).to(device)
    noise_real = torch.randn(2, 3, 4, 5, 8, device=device)
    noise_complex = torch.complex(noise_real, torch.randn_like(noise_real))
    norm_real = prop._normalize_explicit_noise(noise_real, torch.complex64)
    norm_complex = prop._normalize_explicit_noise(noise_complex, torch.complex64)
    imag_real = float(norm_real.imag.abs().max().item())
    complex_imag = float(norm_complex.imag.abs().mean().item())

    seq_shape = (2, 3, 4, 5, 8)
    dt_seq = torch.full((2, 3), 0.5, device=device)
    h_state_seq = torch.ones(seq_shape, device=device, dtype=torch.complex64)
    seq_noise = torch.complex(
        torch.randn(seq_shape, device=device),
        torch.randn(seq_shape, device=device),
    )
    stochastic_seq = prop.generate_stochastic_term_seq(
        seq_shape,
        dt_seq,
        torch.complex64,
        h_state_seq,
        noise_seq=seq_noise,
    )
    seq_zero = prop.generate_stochastic_term_seq(
        seq_shape,
        dt_seq,
        torch.complex64,
        h_state_seq,
        noise_seq=torch.zeros(seq_shape, device=device, dtype=torch.complex64),
    )

    step_shape = (2, 4, 5, 8)
    dt_step = torch.full((2,), 0.5, device=device)
    h_state_step = torch.ones(step_shape, device=device, dtype=torch.complex64)
    step_noise = torch.complex(
        torch.randn(step_shape, device=device),
        torch.randn(step_shape, device=device),
    )
    stochastic_step = prop.generate_stochastic_term_step(
        step_shape,
        dt_step,
        torch.complex64,
        h_state_step,
        noise_step=step_noise,
    )
    step_zero = prop.generate_stochastic_term_step(
        step_shape,
        dt_step,
        torch.complex64,
        h_state_step,
        noise_step=torch.zeros(step_shape, device=device, dtype=torch.complex64),
    )

    passed = (
        all(scales_a[index] < scales_a[index + 1] for index in range(3))
        and all(scales_b[index] > scales_b[index + 1] for index in range(3))
        and err_c < 1e-5
        and imag_real == 0.0
        and complex_imag > 1e-3
        and float(seq_zero.abs().max().item()) == 0.0
        and float(step_zero.abs().max().item()) == 0.0
        and float(stochastic_seq.abs().mean().item()) > 1e-3
        and float(stochastic_step.abs().mean().item()) > 1e-3
    )
    max_err = max(err_c, imag_real)
    detail = (
        f"err_c={err_c:.2e} imag_real={imag_real:.2e} "
        f"complex_imag={complex_imag:.2e} seq_mean={float(stochastic_seq.abs().mean().item()):.2e} "
        f"step_mean={float(stochastic_step.abs().mean().item()):.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
