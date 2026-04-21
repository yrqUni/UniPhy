import sys
from pathlib import Path

from Check.utils import write_result

import torch
from Model.UniPhy.UniPhyOps import ComplexSVDTransform

TEST_ID = "T11"


def run_case(basis, x, dtype):
    w, w_inv = basis.get_matrix(dtype)
    h = basis.encode_with(x, w)
    x_rec = basis.decode_with(h, w_inv)
    return float((x - x_rec).abs().max().item())


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    basis = ComplexSVDTransform(dim=32).to(device)
    x2 = torch.randn(4, 32, device=device)
    x2 = torch.complex(x2, torch.randn_like(x2))
    x4 = torch.randn(2, 3, 5, 32, device=device)
    x4 = torch.complex(x4, torch.randn_like(x4))
    x5 = torch.randn(2, 4, 3, 5, 32, device=device)
    x5 = torch.complex(x5, torch.randn_like(x5))
    with torch.no_grad():
        err_a = max(
            run_case(basis, x2, torch.complex64),
            run_case(basis, x4, torch.complex64),
            run_case(basis, x5, torch.complex64),
        )
        basis.w_re.copy_(torch.randn(32, 32, device=device) * 0.01)
        basis.w_im.copy_(torch.randn(32, 32, device=device) * 0.01)
        err_b = run_case(basis, x2, torch.complex64)
        basis.alpha_logit.copy_(torch.tensor(10.0, device=device))
        w, w_inv = basis.get_matrix(torch.complex128)
        identity = torch.eye(32, dtype=torch.complex128, device=device)
        err_c = float((w @ w_inv - identity).abs().max().item())
    max_err = max(err_a, err_b, err_c)
    status = "PASS" if err_a < 1e-5 and err_b < 1e-4 and err_c < 1e-5 else "FAIL"
    detail = f"err_a={err_a:.2e} err_b={err_b:.2e} err_c={err_c:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
