import torch
from PScan import PScanTriton

def pscan_ref(A, X):
    B, L, C, R = X.shape
    if A.ndim == 4:
        A_mat = torch.diag_embed(A)
    else:
        A_mat = A
    
    Y = torch.zeros_like(X)
    curr = torch.zeros((B, C, R, 1), device=X.device, dtype=X.dtype)
    
    for t in range(L):
        At = A_mat[:, t]
        Xt = X[:, t].unsqueeze(-1)
        curr = torch.matmul(At, curr) + Xt
        Y[:, t] = curr.squeeze(-1)
    return Y

def check_consistency():
    B, L, C, R = 2, 32, 4, 16
    device = "cuda"
    dtype = torch.complex64
    
    A = torch.randn(B, L, C, R, R, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(B, L, C, R, device=device, dtype=dtype, requires_grad=True)
    
    pscan_triton = PScanTriton()
    
    y_triton = pscan_triton(A, X)
    y_ref = pscan_ref(A, X)
    
    fw_err = torch.max(torch.abs(y_triton - y_ref))
    print(f"Forward Consistency Check: {'PASS' if fw_err < 1e-4 else 'FAIL'} (Max Err: {fw_err:.2e})")
    
    grad = torch.randn_like(y_triton)
    y_triton.backward(grad)
    dA_tri, dX_tri = A.grad.clone(), X.grad.clone()
    
    A.grad, X.grad = None, None
    y_ref.backward(grad)
    dA_ref, dX_ref = A.grad.clone(), X.grad.clone()
    
    dx_err = torch.max(torch.abs(dX_tri - dX_ref))
    da_err = torch.max(torch.abs(dA_tri - dA_ref))
    
    print(f"Backward X Consistency Check: {'PASS' if dx_err < 1e-4 else 'FAIL'} (Max Err: {dx_err:.2e})")
    print(f"Backward A Consistency Check: {'PASS' if da_err < 1e-4 else 'FAIL'} (Max Err: {da_err:.2e})")

if __name__ == "__main__":
    check_consistency()
    