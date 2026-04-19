import sys

import torch

from Model.UniPhy.ModelUniPhy import complex_dtype_for
from dt_check.utils import make_tiny_model, write_result

TEST_ID = "T15_dt_zero_mask"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = make_tiny_model(device)
    model.eval()
    batch_size = 2
    with torch.no_grad():
        x = torch.randn(batch_size, 1, 4, 721, 1440, device=device)
        dt_zero = torch.zeros(batch_size, 1, device=device)
        out = model.forward(x, dt_zero)
        out_real = out.real if out.is_complex() else out
        err_a = float((out_real - x).abs().max().item())
        x4 = torch.randn(batch_size, 4, 4, 721, 1440, device=device)
        dt_mixed = torch.tensor([[0.0, 6.0, 0.0, 6.0]] * batch_size, device=device)
        out4 = model.forward(x4, dt_mixed)
        out4_real = out4.real if out4.is_complex() else out4
        err_t0 = float((out4_real[:, 0] - x4[:, 0]).abs().max().item())
        err_t2 = float((out4_real[:, 2] - x4[:, 2]).abs().max().item())
        diff_t1 = float((out4_real[:, 1] - x4[:, 1]).abs().max().item())
        x1 = torch.randn(batch_size, 1, 4, 721, 1440, device=device)
        z_curr = model.encoder(x1)[:, 0]
        states = model._init_states(
            batch_size, device, complex_dtype_for(z_curr.dtype)
        )
        h_prev, flux_prev = states[0]
        dt_step = torch.zeros(batch_size, device=device)
        z_next, _, _ = model.blocks[0].forward_step(
            z_curr, h_prev, dt_step, flux_prev
        )
        err_c = float((z_next - z_curr).abs().max().item())
    max_err = max(err_a, err_t0, err_t2, err_c)
    passed = (
        err_a < 1e-6
        and err_t0 < 1e-6
        and err_t2 < 1e-6
        and diff_t1 > 1e-3
        and err_c < 1e-6
    )
    status = "PASS" if passed else "FAIL"
    detail = (
        f"err_A={err_a:.2e} err_t0={err_t0:.2e} "
        f"err_t2={err_t2:.2e} diff_t1={diff_t1:.2e} err_C={err_c:.2e}"
    )
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
