import torch
from PScan import PScanTriton

def manual_scan_diagonal(A, X):
    A_c = torch.view_as_complex(A)
    X_c = torch.view_as_complex(X)
    B, L, C = A_c.shape
    H = torch.zeros(B, C, device=A.device, dtype=A_c.dtype)
    Y = []
    for t in range(L):
        At = A_c[:, t, :]
        Xt = X_c[:, t, :]
        H = At * H + Xt
        Y.append(H)
    Y_stack = torch.stack(Y, dim=1)
    return torch.view_as_real(Y_stack)

def manual_scan_matrix(A, X):
    B, L, C, D, _ = A.shape
    H = torch.zeros(B, C, D, device=A.device, dtype=A.dtype)
    Y = []
    for t in range(L):
        At = A[:, t, :, :, :]
        Xt = X[:, t, :, :]
        H = torch.einsum('bcd,bd->bc', At, H) + Xt
        Y.append(H)
    return torch.stack(Y, dim=1)

def check_diagonal_mode():
    print("Checking Diagonal Complex Mode...")
    torch.manual_seed(42)
    B, L, C = 2, 64, 4
    
    A = torch.randn(B, L, C, 2, device='cuda') * 0.5 
    X = torch.randn(B, L, C, 2, device='cuda')
    
    A.requires_grad = True
    X.requires_grad = True
    
    pscan = PScanTriton()
    Y_triton = pscan(A, X)
    loss_triton = Y_triton.sum()
    loss_triton.backward()
    grad_A_triton = A.grad.clone()
    grad_X_triton = X.grad.clone()
    
    A.grad = None
    X.grad = None
    
    Y_ref = manual_scan_diagonal(A, X)
    loss_ref = Y_ref.sum()
    loss_ref.backward()
    grad_A_ref = A.grad.clone()
    grad_X_ref = X.grad.clone()
    
    diff = (Y_triton - Y_ref).abs().max().item()
    diff_grad_A = (grad_A_triton - grad_A_ref).abs().max().item()
    diff_grad_X = (grad_X_triton - grad_X_ref).abs().max().item()
    
    print(f"Forward Max Diff: {diff:.6e}")
    print(f"Grad A Max Diff:  {diff_grad_A:.6e}")
    print(f"Grad X Max Diff:  {diff_grad_X:.6e}")
    
    assert diff < 1e-4
    assert diff_grad_A < 1e-4
    assert diff_grad_X < 1e-4
    print("Diagonal Mode Passed!\n")

def check_matrix_mode():
    print("Checking Matrix Real Mode...")
    torch.manual_seed(42)
    B, L, C, D = 2, 64, 4, 8
    
    A = torch.randn(B, L, C, D, D, device='cuda') * 0.1
    X = torch.randn(B, L, C, D, device='cuda')
    
    A.requires_grad = True
    X.requires_grad = True
    
    pscan = PScanTriton()
    Y_triton = pscan(A, X)
    loss_triton = Y_triton.sum()
    loss_triton.backward()
    grad_A_triton = A.grad.clone()
    grad_X_triton = X.grad.clone()
    
    A.grad = None
    X.grad = None
    
    Y_ref = manual_scan_matrix(A, X)
    loss_ref = Y_ref.sum()
    loss_ref.backward()
    grad_A_ref = A.grad.clone()
    grad_X_ref = X.grad.clone()
    
    diff = (Y_triton - Y_ref).abs().max().item()
    diff_grad_A = (grad_A_triton - grad_A_ref).abs().max().item()
    diff_grad_X = (grad_X_triton - grad_X_ref).abs().max().item()
    
    print(f"Forward Max Diff: {diff:.6e}")
    print(f"Grad A Max Diff:  {diff_grad_A:.6e}")
    print(f"Grad X Max Diff:  {diff_grad_X:.6e}")
    
    assert diff < 1e-4
    assert diff_grad_A < 1e-3
    assert diff_grad_X < 1e-3
    print("Matrix Mode Passed!\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        check_diagonal_mode()
        check_matrix_mode()
        print("All checks passed successfully.")
    else:
        print("CUDA not available.")
        