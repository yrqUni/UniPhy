import sys

import torch

from Check.utils import max_diff, write_result
from Model.UniPhy.UniPhyOps import ComplexSVDTransform

TEST_ID = "T06"


def run_case(basis, x, dtype):
    basis_w, basis_w_inv = basis.get_matrix(dtype)
    h = basis.encode_with(x, basis_w)
    x_rec = basis.decode_with(h, basis_w_inv)
    return max_diff(x, x_rec)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    basis = ComplexSVDTransform(dim=32).to(device)
    x2 = torch.complex(
        torch.randn(4, 32, device=device),
        torch.randn(4, 32, device=device),
    )
    x4 = torch.complex(
        torch.randn(2, 3, 5, 32, device=device),
        torch.randn(2, 3, 5, 32, device=device),
    )
    x5 = torch.complex(
        torch.randn(2, 4, 3, 5, 32, device=device),
        torch.randn(2, 4, 3, 5, 32, device=device),
    )
    with torch.no_grad():
        err_roundtrip = max(
            run_case(basis, x2, torch.complex64),
            run_case(basis, x4, torch.complex64),
            run_case(basis, x5, torch.complex64),
        )
        basis.w_re.copy_(torch.randn_like(basis.w_re))
        basis.w_im.copy_(torch.randn_like(basis.w_im))
        basis_w, basis_w_inv = basis.get_matrix(torch.complex128)
        identity = torch.eye(32, dtype=torch.complex128, device=device)
        err_left = max_diff(basis_w @ basis_w_inv, identity)
        err_right = max_diff(basis_w_inv @ basis_w, identity)
    max_err = max(err_roundtrip, err_left, err_right)
    passed = err_roundtrip < 1e-5 and err_left < 1e-2 and err_right < 1e-2
    detail = (
        f"err_roundtrip={err_roundtrip:.2e} err_left={err_left:.2e} "
        f"err_right={err_right:.2e}"
    )
    return ("PASS" if passed else "FAIL"), max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
