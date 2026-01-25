import torch
from PScan import PScanTriton

def manual_pscan_ref(A, X):
    B, L = X.shape[0], X.shape[1]
    is_matrix = A.ndim == X.ndim + 1
    Y = torch.zeros_like(X)
    h = torch.zeros_like(X[:, 0])
    for t in range(L):
        if is_matrix:
            h = torch.einsum('bxy,by->bx', A[:, t], h) + X[:, t]
        else:
            h = A[:, t] * h + X[:, t]
        Y[:, t] = h
    return Y

def check_consistency():
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    pscan = PScanTriton()
    
    B, L, D = 2, 128, 16
    
    print(f"Checking Diagonal Mode (B={B}, L={L}, D={D})...")
    A_diag = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    X_diag = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    
    Y_triton = pscan(A_diag, X_diag)
    Y_ref = manual_pscan_ref(A_diag, X_diag)
    
    err = (Y_triton - Y_ref).abs().max().item()
    print(f"Diagonal Forward Max Error: {err:.2e}")
    if err > 1e-4: raise ValueError("Diagonal Forward Failed")
    
    loss = Y_triton.sum().abs()
    loss.backward()
    grad_A_triton = A_diag.grad.clone()
    grad_X_triton = X_diag.grad.clone()
    
    A_diag.grad.zero_()
    X_diag.grad.zero_()
    
    Y_ref = manual_pscan_ref(A_diag, X_diag)
    loss_ref = Y_ref.sum().abs()
    loss_ref.backward()
    
    err_dA = (grad_A_triton - A_diag.grad).abs().max().item()
    err_dX = (grad_X_triton - X_diag.grad).abs().max().item()
    print(f"Diagonal Backward Max Error dA: {err_dA:.2e}, dX: {err_dX:.2e}")
    if max(err_dA, err_dX) > 1e-4: raise ValueError("Diagonal Backward Failed")

    print("-" * 40)
    print(f"Checking Matrix Mode (B={B}, L={L}, D={D})...")
    A_mat = torch.randn(B, L, D, D, dtype=torch.complex64, device=device, requires_grad=True)
    X_mat = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    
    Y_triton = pscan(A_mat, X_mat)
    Y_ref = manual_pscan_ref(A_mat, X_mat)
    
    err = (Y_triton - Y_ref).abs().max().item()
    print(f"Matrix Forward Max Error: {err:.2e}")
    if err > 1e-4: raise ValueError("Matrix Forward Failed")

    loss = Y_triton.real.sum() + Y_triton.imag.sum()
    loss.backward()
    grad_A_triton = A_mat.grad.clone()
    grad_X_triton = X_mat.grad.clone()
    
    A_mat.grad.zero_()
    X_mat.grad.zero_()
    
    Y_ref = manual_pscan_ref(A_mat, X_mat)
    loss_ref = Y_ref.real.sum() + Y_ref.imag.sum()
    loss_ref.backward()
    
    err_dA = (grad_A_triton - A_mat.grad).abs().max().item()
    err_dX = (grad_X_triton - X_mat.grad).abs().max().item()
    print(f"Matrix Backward Max Error dA: {err_dA:.2e}, dX: {err_dX:.2e}")
    if max(err_dA, err_dX) > 1e-4: raise ValueError("Matrix Backward Failed")
    
    print("-" * 40)
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    check_consistency()
    