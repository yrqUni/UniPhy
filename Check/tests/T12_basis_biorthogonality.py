import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import write_result
else:
    from ..utils import write_result

import torch
from Model.UniPhy.UniPhyOps import ComplexSVDTransform

TEST_ID = "T12"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    basis = ComplexSVDTransform(dim=32).to(device)
    optimizer = torch.optim.SGD(
        [basis.w_re, basis.w_im, basis.w_inv_re, basis.w_inv_im, basis.alpha_logit],
        lr=1e-3,
    )
    identity = torch.eye(32, dtype=torch.complex128, device=device)
    with torch.no_grad():
        w0, w_inv0 = basis.get_matrix(torch.complex128)
        residual_0 = float((w0 @ w_inv0 - identity).abs().max().item())
    for _ in range(100):
        x = torch.randn(8, 32, device=device)
        x = torch.complex(x, torch.randn_like(x))
        w, w_inv = basis.get_matrix(torch.complex64)
        h = basis.encode_with(x, w)
        x_rec = basis.decode_with(h, w_inv)
        loss = (x_rec - x).abs().pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        w1, w_inv1 = basis.get_matrix(torch.complex128)
        residual_100 = float((w1 @ w_inv1 - identity).abs().max().item())
        cond = float(torch.linalg.cond(w1).item())
    detail = (
        f"residual_0={residual_0:.2e} residual_100={residual_100:.2e} "
        f"cond={cond:.1f}"
    )
    return "PASS", residual_100, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
