import sys

import torch

from Check.utils import max_diff, write_result
from Model.UniPhy.PScan import pscan, pscan_torch_tree, pscan_triton

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
        current = torch.einsum("bcij,bcjk->bcik", a_mat[:, step], rows[-1]) + x[:, step]
        rows.append(current)
    return torch.stack(rows, dim=1)


def check_backend(name, fn, a_diag, x_diag, a_mat, x_mat):
    y_diag_ref = sequential_pscan_diag(a_diag, x_diag)
    y_diag = fn(a_diag, x_diag)
    err_diag_forward = max_diff(y_diag_ref, y_diag)

    y_mat_ref = sequential_pscan_mat(a_mat, x_mat)
    y_mat = fn(a_mat, x_mat)
    err_mat_forward = max_diff(y_mat_ref, y_mat)

    a_diag_grad = a_diag.clone().requires_grad_(True)
    x_diag_grad = x_diag.clone().requires_grad_(True)
    a_diag_ref_grad = a_diag.clone().requires_grad_(True)
    x_diag_ref_grad = x_diag.clone().requires_grad_(True)
    sequential_pscan_diag(
        a_diag_ref_grad,
        x_diag_ref_grad,
    ).abs().pow(2).sum().backward()
    fn(a_diag_grad, x_diag_grad).abs().pow(2).sum().backward()
    err_diag_grad_a = max_diff(a_diag_ref_grad.grad, a_diag_grad.grad)
    err_diag_grad_x = max_diff(x_diag_ref_grad.grad, x_diag_grad.grad)

    a_mat_grad = a_mat.clone().requires_grad_(True)
    x_mat_grad = x_mat.clone().requires_grad_(True)
    a_mat_ref_grad = a_mat.clone().requires_grad_(True)
    x_mat_ref_grad = x_mat.clone().requires_grad_(True)
    sequential_pscan_mat(
        a_mat_ref_grad,
        x_mat_ref_grad,
    ).abs().pow(2).sum().backward()
    fn(a_mat_grad, x_mat_grad).abs().pow(2).sum().backward()
    err_mat_grad_a = max_diff(a_mat_ref_grad.grad, a_mat_grad.grad)
    err_mat_grad_x = max_diff(x_mat_ref_grad.grad, x_mat_grad.grad)

    max_err = max(
        err_diag_forward,
        err_mat_forward,
        err_diag_grad_a,
        err_diag_grad_x,
        err_mat_grad_a,
        err_mat_grad_x,
    )
    passed = (
        err_diag_forward < 1e-4
        and err_mat_forward < 1e-4
        and err_diag_grad_a < 1e-3
        and err_diag_grad_x < 1e-3
        and err_mat_grad_a < 1e-3
        and err_mat_grad_x < 1e-3
    )
    detail = (
        f"{name}: diag_forward={err_diag_forward:.2e} "
        f"mat_forward={err_mat_forward:.2e} "
        f"diag_grad_a={err_diag_grad_a:.2e} "
        f"diag_grad_x={err_diag_grad_x:.2e} "
        f"mat_grad_a={err_mat_grad_a:.2e} "
        f"mat_grad_x={err_mat_grad_x:.2e}"
    )
    return passed, max_err, detail


def run_on_device(device):
    torch.manual_seed(42)

    a_diag = torch.randn(2, 16, 4, 2, dtype=torch.complex64, device=device) * 0.5
    x_diag = torch.randn(2, 16, 4, 2, 1, dtype=torch.complex64, device=device)
    a_mat = torch.randn(2, 16, 4, 2, 2, dtype=torch.complex64, device=device) * 0.3
    x_mat = torch.randn(2, 16, 4, 2, 1, dtype=torch.complex64, device=device)
    checks = [
        check_backend(
            "auto",
            pscan,
            a_diag,
            x_diag,
            a_mat,
            x_mat,
        ),
        check_backend(
            "torch_tree",
            pscan_torch_tree,
            a_diag,
            x_diag,
            a_mat,
            x_mat,
        ),
    ]
    triton_status = "skipped:cpu"
    if device.type == "cuda":
        try:
            checks.append(
                check_backend(
                    "triton",
                    pscan_triton,
                    a_diag,
                    x_diag,
                    a_mat,
                    x_mat,
                )
            )
            triton_status = "tested"
        except RuntimeError as exc:
            triton_status = f"unavailable:{exc}"

    x_4d = torch.randn(2, 16, 4, 2, dtype=torch.complex64, device=device)
    y_4d = pscan(a_diag, x_4d)
    shape_ok = y_4d.shape == x_4d.shape

    invalid_cases = [
        (
            torch.randn(2, 16, 4, device=device, dtype=torch.complex64),
            x_diag,
        ),
        (
            a_diag,
            torch.randn(2, 16, 4, device=device, dtype=torch.complex64),
        ),
        (
            torch.randn(2, 16, 4, 2, 3, device=device, dtype=torch.complex64),
            x_mat,
        ),
    ]
    invalid_ok = 0
    for a_invalid, x_invalid in invalid_cases:
        try:
            pscan(a_invalid, x_invalid)
        except ValueError:
            invalid_ok += 1
    invalid_passed = invalid_ok == len(invalid_cases)

    backend_passed = all(entry[0] for entry in checks)
    max_err = max(entry[1] for entry in checks)
    passed = backend_passed and shape_ok and invalid_passed
    backend_detail = " | ".join(entry[2] for entry in checks)
    detail = (
        f"device={device.type} | {backend_detail} | triton={triton_status} "
        f"shape_ok={shape_ok} invalid_cases={invalid_ok}/{len(invalid_cases)}"
    )
    return passed, max_err, detail


def run():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    checks = [run_on_device(device) for device in devices]
    passed = all(entry[0] for entry in checks)
    max_err = max(entry[1] for entry in checks)
    detail = " || ".join(entry[2] for entry in checks)
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
