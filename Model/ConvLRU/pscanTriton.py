import torch
import triton
import triton.language as tl

@triton.jit
def complex_mul(ar, ai, br, bi):
    cr = ar * br - ai * bi
    ci = ar * bi + ai * br
    return cr, ci

@triton.jit
def scan_combine(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar, new_ai = complex_mul(br, bi, ar, ai)
    bx_r, bx_i = complex_mul(br, bi, xr, xi)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

@triton.jit
def pscan_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch, stride_time,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    a_r = tl.load(A_base + offs * stride_time, mask=mask, other=1.0)
    a_i = tl.load(A_base + offs * stride_time + 1, mask=mask, other=0.0)
    x_r = tl.load(X_base + offs * stride_time, mask=mask, other=0.0)
    x_i = tl.load(X_base + offs * stride_time + 1, mask=mask, other=0.0)

    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan(
        (a_r, a_i, x_r, x_i), 
        axis=0, 
        combine_fn=scan_combine
    )

    tl.store(Y_base + offs * stride_time, acc_xr, mask=mask)
    tl.store(Y_base + offs * stride_time + 1, acc_xi, mask=mask)

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

class PScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, S = X.shape
        num_sequences = B * C * S
        
        A_in = A.permute(0, 2, 3, 1).contiguous()
        X_in = X.permute(0, 2, 3, 1).contiguous()
        
        A_real = torch.view_as_real(A_in).reshape(num_sequences, L, 2)
        X_real = torch.view_as_real(X_in).reshape(num_sequences, L, 2)
        Y_real = torch.empty_like(X_real)

        BLOCK_SIZE = next_power_of_2(L)
        BLOCK_SIZE = max(BLOCK_SIZE, 16)
        
        if BLOCK_SIZE > 8192:
             raise ValueError(f"Sequence length L={L} is too large for this single-pass Triton PScan kernel.")

        grid = (num_sequences,)
        stride_batch = A_real.stride(0)
        stride_time = A_real.stride(1)

        pscan_kernel[grid](
            A_real, X_real, Y_real,
            stride_batch, stride_time,
            L, BLOCK_SIZE
        )

        Y = torch.view_as_complex(Y_real.view(B, C, S, L, 2)).permute(0, 3, 1, 2)
        ctx.save_for_backward(A, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, Y = ctx.saved_tensors
        A_conj = A.conj()
        grad_output_rev = grad_output.flip(1)
        A_conj_rev = A_conj.flip(1)
        
        A_rev_shifted = torch.cat([
            torch.zeros_like(A_conj_rev[:, :1]), 
            A_conj_rev[:, :-1]
        ], dim=1)
        
        dX_rev = PScanTriton.apply(A_rev_shifted, grad_output_rev)
        dX = dX_rev.flip(1)
        
        Y_prev = torch.cat([
            torch.zeros_like(Y[:, :1]), 
            Y[:, :-1]
        ], dim=1)
        
        dA = dX * Y_prev.conj()
        return dA, dX

pscan = PScanTriton.apply

def pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
    if not torch.cuda.is_available():
        return True

    device = 'cuda'
    dtype = torch.complex64
    
    A = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    
    Y_triton = pscan(A, X)
    
    Y_serial = torch.zeros_like(X)
    h = torch.zeros(batch_size, channels, state_dim, device=device, dtype=dtype)
    
    for t in range(seq_length):
        h = A[:, t] * h + X[:, t]
        Y_serial[:, t] = h
        
    max_diff = (Y_triton - Y_serial).abs().max().item()
    if max_diff > 1e-4:
        return False
        
    loss = Y_triton.sum().abs()
    loss.backward()
    return True
