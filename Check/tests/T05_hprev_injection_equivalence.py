import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import complex_randn, write_result
else:
    from ..utils import complex_randn, write_result

import torch
from Model.UniPhy.PScan import pscan

TEST_ID = "T05"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    batch_size, steps, height, width, dim = 2, 8, 4, 4, 8
    op_decay = 0.2 * complex_randn((batch_size, steps, height, width, dim), device)
    u_t = complex_randn((batch_size, steps, height, width, dim), device)
    h_prev = complex_randn((batch_size, height, width, dim), device)
    u_t_a = u_t.clone()
    u_t_a[:, 0] = u_t_a[:, 0] + h_prev * op_decay[:, 0]
    a_scan = op_decay.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    x_scan = u_t_a.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim, 1
    )
    out_a = pscan(a_scan, x_scan)
    out_a = out_a.reshape(batch_size, height, width, steps, dim).permute(0, 3, 1, 2, 4)
    out_b = torch.zeros(
        batch_size, steps, height, width, dim, device=device, dtype=torch.complex64
    )
    h = h_prev.reshape(batch_size * height * width, dim)
    op_d = op_decay.permute(0, 2, 3, 1, 4).reshape(
        batch_size * height * width, steps, dim
    )
    u = u_t.permute(0, 2, 3, 1, 4).reshape(batch_size * height * width, steps, dim)
    for t in range(steps):
        h = op_d[:, t] * h + u[:, t]
        out_b[:, t] = h.reshape(batch_size, height, width, dim)
    err = float((out_a - out_b).abs().max().item())
    status = "PASS" if err < 1e-5 else "FAIL"
    detail = f"max_err={err:.2e}"
    return status, err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
