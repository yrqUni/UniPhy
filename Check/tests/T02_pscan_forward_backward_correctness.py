import sys

import torch

from Check.utils import max_diff, write_result
from Model.UniPhy.PScan import pscan

TEST_ID = "T02"


def sequential_pscan_diag(a_diag, x):
    batch_size, steps, channels, dim, _ = x.shape
    a_mat = torch.zeros(
        batch_size,
        steps,
        channels,
        dim,
        dim,
        dtype=a_diag.dtype,
        device=a_diag.device,
    )
    idx = torch.arange(dim, device=a_diag.device)
    a_mat[..., idx, idx] = a_diag
    return sequential_pscan_mat(a_mat, x)


def sequential_pscan_mat(a_mat, x):
    rows = [x[:, 0]]
    for step in range(1, x.shape[1]):
        current = (
            torch.einsum("bcij,bcjk->bcik", a_mat[:, step], rows[-1])
            + x[:, step]
        )
        rows.append(current)
    return torch.stack(rows, dim=1)


def run():
    if not torch.cuda.is_available():
        return "SKIP", "-", "cuda_required"
    device = torch.device("cuda")
    torch.manual_seed(42)

    a_diag = torch.randn(2, 16, 4, 2, dtype=torch.complex64, device=device) * 0.5
    x_diag = torch.randn(2, 16, 4, 2, 1, dtype=torch.complex64, device=device)
    y_diag_ref = sequential_pscan_diag(a_diag, x_diag)
    y_diag = pscan(a_diag, x_diag)
    err_diag = max_diff(y_diag_ref, y_diag)

    a_mat = torch.randn(2, 16, 4, 2, 2, dtype=torch.complex64, device=device) * 0.3
    x_mat = torch.randn(2, 16, 4, 2, 1, dtype=torch.complex64, device=device)
    y_mat_ref = sequential_pscan_mat(a_mat, x_mat)
    y_mat = pscan(a_mat, x_mat)
    err_mat = max_diff(y_mat_ref, y_mat)

    a_diag_grad = a_diag.clone().requires_grad_(True)
    x_diag_grad = x_diag.clone().requires_grad_(True)
    a_diag_ref_grad = a_diag.clone().requires_grad_(True)
    x_diag_ref_grad = x_diag.clone().requires_grad_(True)
    sequential_pscan_diag(
        a_diag_ref_grad,
        x_diag_ref_grad,
    ).abs().pow(2).sum().backward()
    pscan(a_diag_grad, x_diag_grad).abs().pow(2).sum().backward()
    err_diag_grad = max(
        max_diff(a_diag_ref_grad.grad, a_diag_grad.grad),
        max_diff(x_diag_ref_grad.grad, x_diag_grad.grad),
    )

    a_mat_grad = a_mat.clone().requires_grad_(True)
    x_mat_grad = x_mat.clone().requires_grad_(True)
    a_mat_ref_grad = a_mat.clone().requires_grad_(True)
    x_mat_ref_grad = x_mat.clone().requires_grad_(True)
    sequential_pscan_mat(
        a_mat_ref_grad,
        x_mat_ref_grad,
    ).abs().pow(2).sum().backward()
    pscan(a_mat_grad, x_mat_grad).abs().pow(2).sum().backward()
    err_mat_grad = max(
        max_diff(a_mat_ref_grad.grad, a_mat_grad.grad),
        max_diff(x_mat_ref_grad.grad, x_mat_grad.grad),
    )

    x_4d = torch.randn(2, 16, 4, 2, dtype=torch.complex64, device=device)
    y_4d = pscan(a_diag, x_4d)
    shape_ok = y_4d.shape == x_4d.shape

    max_err = max(err_diag, err_mat, err_diag_grad, err_mat_grad)
    passed = (
        err_diag < 1e-4
        and err_mat < 1e-4
        and err_diag_grad < 1e-3
        and err_mat_grad < 1e-3
        and shape_ok
    )
    detail = (
        f"err_diag={err_diag:.2e} err_mat={err_mat:.2e} "
        f"err_diag_grad={err_diag_grad:.2e} err_mat_grad={err_mat_grad:.2e} "
        f"shape_ok={shape_ok}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
