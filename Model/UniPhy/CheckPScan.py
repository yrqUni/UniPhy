import torch
import torch.nn as nn
from PScan import PScanTriton

def manual_pscan(A, X):
    B, L, D = X.shape
    is_mat = A.ndim == X.ndim + 1
    Y = torch.zeros_like(X)
    h = torch.zeros_like(X[:, 0])
    for t in range(L):
        if is_mat:
            h = torch.einsum('bij,bj->bi', A[:, t], h) + X[:, t]
        else:
            h = A[:, t] * h + X[:, t]
        Y[:, t] = h
    return Y

def check():
    device = torch.device('cuda')
    pscan = PScanTriton()
    B, L, D = 2, 128, 16
    
    print("Checking Diagonal Mode...")
    A = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    X = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Y_ref = manual_pscan(A, X)
    Y_tri = pscan(A, X)
    print(f"Forward Err: {(Y_tri - Y_ref).abs().max():.2e}")
    
    Y_tri.abs().sum().backward()
    dA_tri, dX_tri = A.grad.clone(), X.grad.clone()
    A.grad.zero_(); X.grad.zero_()
    Y_ref.abs().sum().backward()
    print(f"Backward dA Err: {(dA_tri - A.grad).abs().max():.2e}")
    print(f"Backward dX Err: {(dX_tri - X.grad).abs().max():.2e}")
    
    print("\nChecking Matrix Mode...")
    A = torch.randn(B, L, D, D, dtype=torch.complex64, device=device, requires_grad=True)
    X = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Y_ref = manual_pscan(A, X)
    Y_tri = pscan(A, X)
    print(f"Forward Err: {(Y_tri - Y_ref).abs().max():.2e}")
    
    Y_tri.abs().sum().backward()
    dA_tri, dX_tri = A.grad.clone(), X.grad.clone()
    A.grad.zero_(); X.grad.zero_()
    Y_ref.abs().sum().backward()
    print(f"Backward dA Err: {(dA_tri - A.grad).abs().max():.2e}")
    print(f"Backward dX Err: {(dX_tri - X.grad).abs().max():.2e}")

if __name__ == "__main__":
    check()
    