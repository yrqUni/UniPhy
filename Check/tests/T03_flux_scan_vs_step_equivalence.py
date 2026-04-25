import sys

import torch

from Check.utils import complex_randn, max_diff, write_result
from Model.UniPhy.PScan import pscan
from Model.UniPhy.UniPhyOps import GlobalFluxTracker

TEST_ID = "T03"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    batch_size, steps, height, width, dim = 2, 8, 4, 4, 8
    op_decay = 0.2 * complex_randn((batch_size, steps, height, width, dim), device)
    forcing = complex_randn((batch_size, steps, height, width, dim), device)
    h_prev = complex_randn((batch_size, height, width, dim), device)
    forcing_scan = forcing.clone()
    forcing_scan[:, 0] = forcing_scan[:, 0] + h_prev * op_decay[:, 0]
    a_scan = op_decay.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    x_scan = forcing_scan.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    out_scan = pscan(a_scan, x_scan)
    out_scan = out_scan.reshape(batch_size, height, width, steps, dim).permute(
        0, 3, 1, 2, 4
    )
    out_ref = torch.zeros_like(out_scan)
    state = h_prev.reshape(batch_size * height * width, dim)
    decay_ref = op_decay.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim
    )
    forcing_ref = forcing.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim
    )
    for step in range(steps):
        state = decay_ref[:, step] * state + forcing_ref[:, step]
        out_ref[:, step] = state.reshape(batch_size, height, width, dim)
    err_state = max_diff(out_scan, out_ref)

    tracker = GlobalFluxTracker(dim=8, dt_ref=1.0).to(device).double()
    x_mean_seq = torch.complex(
        torch.randn(batch_size, 7, 8, device=device, dtype=torch.float64),
        torch.randn(batch_size, 7, 8, device=device, dtype=torch.float64),
    )
    dt_seq = torch.rand(batch_size, 7, device=device, dtype=torch.float64) * 1.5 + 0.1
    flux_prev = tracker.get_initial_state(batch_size, device, torch.complex128)
    scan_decay, scan_forcing = tracker.get_scan_operators(x_mean_seq, dt_seq)
    flux_scan = pscan(scan_decay, scan_forcing).squeeze(-1)
    flux_scan = flux_scan + flux_prev.unsqueeze(1) * torch.cumprod(
        scan_decay.squeeze(-1), dim=1
    )
    rows = []
    flux_state = flux_prev
    for step in range(dt_seq.shape[1]):
        flux_state, _, _ = tracker.forward_step(
            flux_state,
            x_mean_seq[:, step],
            dt_seq[:, step],
        )
        rows.append(flux_state)
    flux_ref = torch.stack(rows, dim=1)
    err_flux = max_diff(flux_scan, flux_ref)

    max_err = max(err_state, err_flux)
    passed = err_state < 1e-5 and err_flux < 1e-10
    detail = f"err_state={err_state:.2e} err_flux={err_flux:.2e}"
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
