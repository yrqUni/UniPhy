import sys
from pathlib import Path

from Check.utils import write_result

import torch
from Model.UniPhy.UniPhyOps import ComplexSVDTransform

TEST_ID = "T18"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    basis = ComplexSVDTransform(dim=32).to(device)
    with torch.no_grad():
        basis.w_re.copy_(torch.randn_like(basis.w_re))
        basis.w_im.copy_(torch.randn_like(basis.w_im))
        w, w_inv = basis.get_matrix(torch.complex128)
        identity = torch.eye(32, dtype=torch.complex128, device=device)
        err_left = float((w @ w_inv - identity).abs().max().item())
        err_right = float((w_inv @ w - identity).abs().max().item())
    max_err = max(err_left, err_right)
    status = "PASS" if max_err < 1e-2 else "FAIL"
    detail = f"err_left={err_left:.2e} err_right={err_right:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
