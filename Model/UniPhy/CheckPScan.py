import torch
import torch.nn.functional as F
from PScan import PScan

def sequential_scan(A, X):
    B, L, C, R, _ = X.shape
    Y = torch.zeros_like(X)
    h = torch.zeros((B, C, R, R), device=X.device, dtype=X.dtype)
    
    is_diag = (A.ndim == 4)
    
    for t in range(L):
        x_t = X[:, t]
        if is_diag:
            a_t = A[:, t]
            h = a_t.unsqueeze(-1) * h + x_t
        else:
            a_t = A[:, t]
            h = torch.matmul(a_t, h) + x_t
        Y[:, t] = h
        
    return Y

def check_consistency():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, L, C, R = 2, 64, 4, 4
    
    print(f"--- Testing Diagonal A (BLCR) ---")
    A_diag = torch.randn(B, L, C, R, device=device, requires_grad=True)
    X = torch.randn(B, L, C, R, R, device=device, requires_grad=True)
    
    pscan = PScan()
    
    Y_triton = pscan(A_diag, X)
    Y_ref = sequential_scan(A_diag, X)
    
    diff = torch.abs(Y_triton - Y_ref).max()
    print(f"Forward Max Difference: {diff.item()}")
    
    grad_output = torch.randn_like(Y_triton)
    Y_triton.backward(grad_output)
    dA_triton, dX_triton = A_diag.grad.clone(), X.grad.clone()
    
    A_diag.grad = None
    X.grad = None
    
    Y_ref = sequential_scan(A_diag, X)
    Y_ref.backward(grad_output)
    dA_ref, dX_ref = A_diag.grad, X.grad
    
    diff_da = torch.abs(dA_triton - dA_ref).max()
    diff_dx = torch.abs(dX_triton - dX_ref).max()
    print(f"Backward dA Diff: {diff_da.item()}")
    print(f"Backward dX Diff: {diff_dx.item()}")
    
    assert diff < 1e-4
    assert diff_da < 1e-4
    assert diff_dx < 1e-4

    print(f"\n--- Testing Matrix A (BLCRR) ---")
    A_mat = torch.randn(B, L, C, R, R, device=device, requires_grad=True) * 0.1
    X = torch.randn(B, L, C, R, R, device=device, requires_grad=True)
    
    Y_triton = pscan(A_mat, X)
    Y_ref = sequential_scan(A_mat, X)
    
    diff = torch.abs(Y_triton - Y_ref).max()
    print(f"Forward Max Difference: {diff.item()}")
    
    grad_output = torch.randn_like(Y_triton)
    Y_triton.backward(grad_output)
    dA_triton, dX_triton = A_mat.grad.clone(), X.grad.clone()
    
    A_mat.grad = None
    X.grad = None
    
    Y_ref = sequential_scan(A_mat, X)
    Y_ref.backward(grad_output)
    dA_ref, dX_ref = A_mat.grad, X.grad
    
    diff_da = torch.abs(dA_triton - dA_ref).max()
    diff_dx = torch.abs(dX_triton - dX_ref).max()
    print(f"Backward dA Diff: {diff_da.item()}")
    print(f"Backward dX Diff: {diff_dx.item()}")

    assert diff < 1e-4
    assert diff_da < 1e-3
    assert diff_dx < 1e-3

if __name__ == "__main__":
    check_consistency()
    