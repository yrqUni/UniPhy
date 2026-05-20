import sys

import torch

from Check.utils import write_result
from Model.UniPhy.UniPhyOps import _etd2_coefficients

TEST_ID = "T14"


def _analytic_linear_u(lam, h0, u0, u1, dt):
    exp_arg = lam * dt
    decay = torch.exp(exp_arg)
    one = torch.ones_like(exp_arg)
    phi1 = (
        torch.where(
            exp_arg.abs() < 1e-14,
            one,
            torch.expm1(exp_arg) / exp_arg,
        )
        * dt
    )
    phi2 = (
        torch.where(
            exp_arg.abs() < 1e-14,
            0.5 * one,
            (torch.expm1(exp_arg) - exp_arg) / (exp_arg * exp_arg),
        )
        * dt
    )
    return decay * h0 + (phi1 - phi2) * u0 + phi2 * u1


def _etd2_constant_u_error_over_full_interval(lam, h0, u_func, T, n_steps):
    dt = T / n_steps
    h = h0
    for n in range(n_steps):
        t_n = n * dt
        u_n = u_func(t_n)
        u_np1 = u_func(t_n + dt)
        decay, alpha, beta = _etd2_coefficients(
            lam,
            torch.full_like(lam.real, dt).to(torch.complex128),
        )
        h = decay * h + alpha * u_n + beta * u_np1
    return h


def _reference_solution(lam, h0, u_func, T, n_fine=200000):
    dt = T / n_fine
    h = h0
    for n in range(n_fine):
        t_n = n * dt
        u_n = u_func(t_n)
        u_np1 = u_func(t_n + dt)
        decay, alpha, beta = _etd2_coefficients(
            lam,
            torch.full_like(lam.real, dt).to(torch.complex128),
        )
        h = decay * h + alpha * u_n + beta * u_np1
    return h


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 4

    lam = torch.complex(
        torch.full((dim,), -0.7, dtype=torch.float64, device=device),
        torch.full((dim,), 0.4, dtype=torch.float64, device=device),
    )

    torch.manual_seed(0)
    h0 = torch.complex(
        torch.randn(dim, dtype=torch.float64, device=device),
        torch.randn(dim, dtype=torch.float64, device=device),
    )
    a_re = torch.randn(dim, dtype=torch.float64, device=device)
    a_im = torch.randn(dim, dtype=torch.float64, device=device)
    b_re = torch.randn(dim, dtype=torch.float64, device=device)
    b_im = torch.randn(dim, dtype=torch.float64, device=device)

    def u_func(t):
        re = a_re * torch.sin(
            torch.tensor(t, dtype=torch.float64, device=device)
        ) + b_re * torch.cos(torch.tensor(2.0 * t, dtype=torch.float64, device=device))
        im = a_im * torch.cos(
            torch.tensor(t, dtype=torch.float64, device=device)
        ) + b_im * torch.sin(torch.tensor(0.7 * t, dtype=torch.float64, device=device))
        return torch.complex(re, im)

    T = 2.0
    h_ref = _reference_solution(lam, h0, u_func, T, n_fine=200000)

    errs = []
    n_steps_list = [16, 32, 64, 128]
    for n in n_steps_list:
        h = _etd2_constant_u_error_over_full_interval(lam, h0, u_func, T, n)
        err = float((h - h_ref).abs().max().item())
        errs.append(err)

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]

    floor = 1e-12
    nontrivial = [(i, r) for i, r in enumerate(ratios) if errs[i] > floor]
    if not nontrivial:
        detail = (
            f"errs={['%.2e' % e for e in errs]} " f"ratios all below floor={floor:.1e}"
        )
        return "PASS", float(max(errs)), detail
    ratios_test = [r for _, r in nontrivial]
    mean_ratio = sum(ratios_test) / len(ratios_test)

    passed = 3.0 < mean_ratio < 5.0
    detail = (
        f"errs={['%.2e' % e for e in errs]} "
        f"ratios={['%.2f' % r for r in ratios]} "
        f"mean_ratio={mean_ratio:.2f} (expect ~4 for 2nd order)"
    )
    return ("PASS" if passed else "FAIL"), float(max(errs)), detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
