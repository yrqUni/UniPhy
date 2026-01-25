import torch
from PScan import PScanTriton

def pscan_ref(A, X):
    B, L, C, R = X.shape
    AM = torch.diag_embed(A) if A.ndim == 4 else A
    Y, curr = torch.zeros_like(X), torch.zeros((B, C, R, 1), device=X.device, dtype=X.dtype)
    for t in range(L):
        curr = torch.matmul(AM[:, t], curr) + X[:, t].unsqueeze(-1)
        Y[:, t] = curr.squeeze(-1)
    return Y

def check_consistency():
    B, L, C, R = 2, 16, 2, 16
    device, dtype = "cuda", torch.complex64
    A = torch.randn(B, L, C, R, R, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(B, L, C, R, device=device, dtype=dtype, requires_grad=True)
    pscan = PScanTriton()
    y_tri = pscan(A, X)
    y_ref = pscan_ref(A, X)
    fw_err = torch.max(torch.abs(y_tri - y_ref))
    print(f"FW: {'PASS' if fw_err < 1e-4 else 'FAIL'} (Err: {fw_err:.2e})")
    grad = torch.randn_like(y_tri)
    y_tri.backward(grad)
    da_tri, dx_tri = A.grad.clone(), X.grad.clone()
    A.grad, X.grad = None, None
    y_ref.backward(grad)
    dx_err, da_err = torch.max(torch.abs(dx_tri - X.grad)), torch.max(torch.abs(da_tri - A.grad))
    print(f"BW dX: {'PASS' if dx_err < 1e-4 else 'FAIL'} (Err: {dx_err:.2e})")
    print(f"BW dA: {'PASS' if da_err < 1e-4 else 'FAIL'} (Err: {da_err:.2e})")

if __name__ == "__main__":
    check_consistency()
    