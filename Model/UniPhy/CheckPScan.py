import torch
from PScan import PScanTriton

def pscan_ref(A, X):
    B, L, C, R, _ = X.shape if X.ndim == 5 else (*X.shape, 1)
    if A.ndim == 4:
        A = torch.diag_embed(A)
    
    Y = torch.zeros_like(X)
    curr = torch.zeros((B, C, R, 1), device=X.device, dtype=X.dtype)
    
    for t in range(L):
        At = A[:, t] # (B, C, R, R)
        Xt = X[:, t].unsqueeze(-1) # (B, C, R, 1)
        curr = torch.matmul(At, curr) + Xt
        Y[:, t] = curr.squeeze(-1)
    return Y

def check():
    B, L, C, R = 2, 64, 4, 16
    device = "cuda"
    dtype = torch.complex64
    
    A = torch.randn(B, L, C, R, R, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(B, L, C, R, device=device, dtype=dtype, requires_grad=True)
    
    pscan_triton = PScanTriton()
    
    # Forward Check
    y_triton = pscan_triton(A, X)
    y_ref = pscan_ref(A, X)
    
    print(f"FW Diff: {torch.norm(y_triton - y_ref).item()}")
    
    # Backward Check
    grad = torch.randn_like(y_triton)
    y_triton.backward(grad)
    
    dA_triton, dX_triton = A.grad.clone(), X.grad.clone()
    A.grad, X.grad = None, None
    
    y_ref.backward(grad)
    dA_ref, dX_ref = A.grad.clone(), X.grad.clone()
    
    print(f"BW dA Diff: {torch.norm(dA_triton - dA_ref).item()}")
    print(f"BW dX Diff: {torch.norm(dX_triton - dX_ref).item()}")

if __name__ == "__main__":
    check()
    