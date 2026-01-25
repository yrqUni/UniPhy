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
    B, L, D = 2, 64, 16 # D 必须为 tl.dot 支持的尺寸（如16, 32）
    
    print(">>> Checking Diagonal Mode...")
    Ad = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Xd = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Yr = manual_pscan(Ad, Xd)
    Yt = pscan(Ad, Xd)
    print(f"Forward Max Err: {(Yt - Yr).abs().max():.2e}")
    
    Yt.abs().sum().backward()
    dAd, dXd = Ad.grad.clone(), Xd.grad.clone()
    Ad.grad.zero_(); Xd.grad.zero_()
    Yr.abs().sum().backward()
    print(f"Backward dA Max Err: {(dAd - Ad.grad).abs().max():.2e}")
    print(f"Backward dX Max Err: {(dXd - Xd.grad).abs().max():.2e}")
    
    print("\n>>> Checking Matrix Mode...")
    Am = torch.randn(B, L, D, D, dtype=torch.complex64, device=device, requires_grad=True)
    Xm = torch.randn(B, L, D, dtype=torch.complex64, device=device, requires_grad=True)
    Yr = manual_pscan(Am, Xm)
    Yt = pscan(Am, Xm)
    print(f"Forward Max Err: {(Yt - Yr).abs().max():.2e}")
    
    Yt.abs().sum().backward()
    dAm, dXm = Am.grad.clone(), Xm.grad.clone()
    Am.grad.zero_(); Xm.grad.zero_()
    Yr.abs().sum().backward()
    print(f"Backward dA Max Err: {(dAm - Am.grad).abs().max():.2e}")
    print(f"Backward dX Max Err: {(dXm - Xm.grad).abs().max():.2e}")

if __name__ == "__main__":
    check()
    