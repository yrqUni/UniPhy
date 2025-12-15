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
        dtype_orig = X_in.dtype
        if dtype_orig == torch.float16 or dtype_orig == torch.bfloat16:
            calc_dtype = torch.float32
        elif dtype_orig == torch.complex32:
            calc_dtype = torch.complex64
        else:
            calc_dtype = dtype_orig
        if L == npo2(L):
            A = A_in.clone().to(dtype=calc_dtype).contiguous()
            X = X_in.clone().to(dtype=calc_dtype).contiguous()
        else:
            A = pad_npo2(A_in).to(dtype=calc_dtype).contiguous()
            X = pad_npo2(X_in).to(dtype=calc_dtype).contiguous()
        PScan.pscan(A, X)
        ctx.save_for_backward(A_in, X)
        ctx.calc_dtype = calc_dtype
        return X[:, :L].to(dtype=dtype_orig)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, H_pad = ctx.saved_tensors
        calc_dtype = ctx.calc_dtype
        L = grad_output_in.size(1)
        if L == npo2(L):
            grad_output = grad_output_in.clone().to(dtype=calc_dtype).contiguous()
            A_pad = A_in.clone().to(dtype=calc_dtype).contiguous()
        else:
            grad_output = pad_npo2(grad_output_in).to(dtype=calc_dtype).contiguous()
            A_pad = pad_npo2(A_in).to(dtype=calc_dtype).contiguous()
        A_shift = F.pad(A_pad[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))
        PScan.pscan_rev(A_shift.conj(), grad_output)
        gradX = grad_output[:, :L]
        gradA_full = torch.zeros_like(A_pad)
        gradA_full[:, 1:, :, :, 0] = (H_pad[:, :-1].conj() * grad_output[:, 1:]).sum(dim=-1)
        gradA = gradA_full[:, :L]
        return gradA.to(dtype=A_in.dtype), gradX.to(dtype=A_in.dtype)

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

def pscan_check(batch_size=2, seq_length=13, channels=4, state_dim=8, device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    print(f"Running PScan Check on {device}...")
    dtype = torch.complex128
    A = torch.randn(batch_size, seq_length, channels, state_dim, 1, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(batch_size, seq_length, channels, state_dim, state_dim, device=device, dtype=dtype, requires_grad=True)
    H_pscan = pscan(A, X)
    H_serial = serial_scan(A, X)
    fwd_diff = (H_pscan - H_serial).abs().max()
    print(f"Forward Max Diff: {fwd_diff:.2e}")
    if fwd_diff > 1e-10:
        return False
    from torch.autograd import gradcheck
    A_small = torch.randn(1, 5, 2, 2, 1, device=device, dtype=dtype, requires_grad=True)
    X_small = torch.randn(1, 5, 2, 2, 2, device=device, dtype=dtype, requires_grad=True)
    if gradcheck(pscan, (A_small, X_small), eps=1e-6, atol=1e-4, rtol=1e-3):
        print("✅ Backward (Gradient) check passed!")
        return True
    return False

if __name__ == "__main__":
    pscan_check()
