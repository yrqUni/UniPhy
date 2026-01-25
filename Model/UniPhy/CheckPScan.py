import torch
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
    A_d = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    X_d = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Y_ref_d = manual_pscan(A_d, X_d)
    Y_tri_d = pscan(A_d, X_d)
    print(f"Forward Err: {(Y_tri_d - Y_ref_d).abs().max():.2e}")
    
    Y_tri_d.abs().sum().backward()
    dA_tri_d, dX_tri_d = A_d.grad.clone(), X_d.grad.clone()
    A_d.grad.zero_(); X_d.grad.zero_()
    Y_ref_d.abs().sum().backward()
    print(f"Backward dA Err: {(dA_tri_d - A_d.grad).abs().max():.2e}")
    print(f"Backward dX Err: {(dX_tri_d - X_d.grad).abs().max():.2e}")
    
    print("\nChecking Matrix Mode...")
    A_m = torch.randn(B, L, D, D, dtype=torch.complex64, device=device, requires_grad=True)
    X_m = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Y_ref_m = manual_pscan(A_m, X_m)
    Y_tri_m = pscan(A_m, X_m)
    print(f"Forward Err: {(Y_tri_m - Y_ref_m).abs().max():.2e}")
    
    Y_tri_m.abs().sum().backward()
    dA_tri_m, dX_tri_m = A_m.grad.clone(), X_m.grad.clone()
    A_m.grad.zero_(); X_m.grad.zero_()
    Y_ref_m.abs().sum().backward()
    print(f"Backward dA Err: {(dA_tri_m - A_m.grad).abs().max():.2e}")
    print(f"Backward dX Err: {(dX_tri_m - X_m.grad).abs().max():.2e}")

if __name__ == "__main__":
    check()
    