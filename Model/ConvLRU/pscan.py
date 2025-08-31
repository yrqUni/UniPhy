import math
import torch
import torch.nn.functional as F

def npo2(length: int) -> int:
    return 1 if length <= 1 else 2 ** math.ceil(math.log2(length))

def pad_npo2(x: torch.Tensor) -> torch.Tensor:
    L = x.size(1)
    L2 = npo2(L)
    if L2 == L:
        return x
    pad_tuple = (0, 0, 0, 0, 0, 0, 0, L2 - L)
    return F.pad(x, pad_tuple, "constant", 0)

class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Aa = Aa[:, :, 1]
            Xa = Xa[:, :, 1]
        if Xa.size(1) == 4:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            Aa[:, 1].mul_(Aa[:, 0])
            Xa[:, 3].add_(Aa[:, 3].mul(Xa[:, 2] + Aa[:, 2].mul(Xa[:, 1])))
        elif Xa.size(1) == 2:
            Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 0]))
            return
        else:
            return
        Aa = A[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa = X[:, 2 ** (num_steps - 2) - 1:L:2 ** (num_steps - 2)]
        Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 1]))
        Aa[:, 2].mul_(Aa[:, 1])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 2 ** k - 1:L:2 ** k]
            Xa = X[:, 2 ** k - 1:L:2 ** k]
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        B, L, C, S, _ = A.size()
        num_steps = int(math.log2(L))
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            Aa[:, :, 0].mul_(Aa[:, :, 1])
            Aa = Aa[:, :, 0]
            Xa = Xa[:, :, 0]
        if Xa.size(1) == 4:
            Xa[:, 2].add_(Aa[:, 2].mul(Xa[:, 3]))
            Aa[:, 2].mul_(Aa[:, 3])
            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1].add(Aa[:, 1].mul(Xa[:, 2]))))
        elif Xa.size(1) == 2:
            Xa[:, 0].add_(Aa[:, 0].mul(Xa[:, 1]))
            return
        else:
            return
        Aa = A[:, 0:L:2 ** (num_steps - 2)]
        Xa = X[:, 0:L:2 ** (num_steps - 2)]
        Xa[:, 1].add_(Aa[:, 1].mul(Xa[:, 2]))
        Aa[:, 1].mul_(Aa[:, 2])
        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, 0:L:2 ** k]
            Xa = X[:, 0:L:2 ** k]
            T = Xa.size(1)
            Aa = Aa.view(B, T // 2, 2, C, S, 1)
            Xa = Xa.view(B, T // 2, 2, C, S, S)
            Xa[:, :-1, 1].add_(Aa[:, :-1, 1].mul(Xa[:, 1:, 0]))
            Aa[:, :-1, 1].mul_(Aa[:, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        return X[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, H_pad = ctx.saved_tensors
        L = grad_output_in.size(1)
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            A_pad = A_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in)
            A_pad = pad_npo2(A_in)
        A_shift = F.pad(A_pad[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))
        PScan.pscan_rev(A_shift, grad_output)
        gradX = grad_output[:, :L]
        gradA_full = torch.zeros_like(A_pad)
        gradA_full[:, 1:, :, :, 0] = (H_pad[:, :-1] * grad_output[:, 1:]).sum(dim=-1)
        gradA = gradA_full[:, :L]
        return gradA, gradX

pscan = PScan.apply

def serial_scan(A, X):
    B, L, C, S, _ = A.size()
    H = torch.zeros_like(X)
    for b in range(B):
        for l in range(L):
            if l == 0:
                H[b, l] = X[b, l].clone()
            else:
                H[b, l] = A[b, l] * H[b, l - 1].clone() + X[b, l].clone()
    return H

def pscan_check(batch_size=2, seq_length=13, channels=8, state_dim=16):
    torch.manual_seed(0)
    A_tensor = torch.rand(batch_size, seq_length, channels, state_dim, 1, dtype=torch.float64)
    A1 = torch.nn.Parameter(A_tensor.clone())
    A2 = torch.nn.Parameter(A_tensor.clone())
    X1 = torch.rand(batch_size, seq_length, channels, state_dim, state_dim, dtype=torch.float64, requires_grad=True)
    X2 = X1.clone().detach().requires_grad_(True)
    H_gt = torch.rand(batch_size, seq_length, channels, state_dim, state_dim, dtype=torch.float64)
    loss_fn = torch.nn.MSELoss()
    H_pscan = PScan.apply(A1, X1)
    loss_pscan = loss_fn(H_pscan, H_gt)
    loss_pscan.backward()
    H_serial = serial_scan(A2, X2)
    loss_serial = loss_fn(H_serial, H_gt)
    loss_serial.backward()
    fwd_ok = torch.allclose(H_pscan, H_serial, rtol=1e-5, atol=1e-7)
    gradA_ok = torch.allclose(A1.grad, A2.grad, rtol=1e-5, atol=1e-7)
    gradX_ok = torch.allclose(X1.grad, X2.grad, rtol=1e-5, atol=1e-7)
    import gc
    del A_tensor, A1, A2, X1, X2, H_gt, H_pscan, H_serial, loss_pscan, loss_serial, loss_fn
    gc.collect()
    torch.cuda.empty_cache()
    return (fwd_ok, gradA_ok and gradX_ok)

assert pscan_check(), "PScan check failed"
