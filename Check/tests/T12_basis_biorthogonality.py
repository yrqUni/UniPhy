import sys
from pathlib import Path

from Check.utils import write_result

import torch
from Model.UniPhy.UniPhyOps import ComplexSVDTransform

TEST_ID = "T12"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    basis = ComplexSVDTransform(dim=32).to(device)
    identity = torch.eye(32, dtype=torch.complex128, device=device)
    with torch.no_grad():
        basis.w_re.copy_(torch.randn_like(basis.w_re))
        basis.w_im.copy_(torch.randn_like(basis.w_im))
        w_good, w_inv_good = basis.get_matrix(torch.complex128)
        residual_good = float((w_good @ w_inv_good - identity).abs().max().item())
        broken_inverse = w_inv_good * 1.25
        residual_broken = float((w_good @ broken_inverse - identity).abs().max().item())
    passed = residual_good < 1e-8 and residual_broken > 1e-2
    max_err = max(
        residual_good,
        0.0 if residual_broken > 1e-2 else 1e-2 - residual_broken,
    )
    status = "PASS" if passed else "FAIL"
    detail = f"residual_good={residual_good:.2e} residual_broken={residual_broken:.2e}"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
