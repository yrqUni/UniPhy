import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@triton.jit
def mat2x2_scan_combine(
    a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
    x00r, x00i, x01r, x01i, x10r, x10i, x11r, x11i,
    b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i,
    y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i,
):
    c00r, c00i = complex_mul(b00r, b00i, a00r, a00i)
    t1r, t1i = complex_mul(b01r, b01i, a10r, a10i)
    c00r = c00r + t1r
    c00i = c00i + t1i

    c01r, c01i = complex_mul(b00r, b00i, a01r, a01i)
    t2r, t2i = complex_mul(b01r, b01i, a11r, a11i)
    c01r = c01r + t2r
    c01i = c01i + t2i

    c10r, c10i = complex_mul(b10r, b10i, a00r, a00i)
    t3r, t3i = complex_mul(b11r, b11i, a10r, a10i)
    c10r = c10r + t3r
    c10i = c10i + t3i

    c11r, c11i = complex_mul(b10r, b10i, a01r, a01i)
    t4r, t4i = complex_mul(b11r, b11i, a11r, a11i)
    c11r = c11r + t4r
    c11i = c11i + t4i

    bx00r, bx00i = complex_mul(b00r, b00i, x00r, x00i)
    bx01r, bx01i = complex_mul(b01r, b01i, x10r, x10i)
    z00r = bx00r + bx01r + y00r
    z00i = bx00i + bx01i + y00i

    bx02r, bx02i = complex_mul(b00r, b00i, x01r, x01i)
    bx03r, bx03i = complex_mul(b01r, b01i, x11r, x11i)
    z01r = bx02r + bx03r + y01r
    z01i = bx02i + bx03i + y01i

    bx10r, bx10i = complex_mul(b10r, b10i, x00r, x00i)
    bx11r, bx11i = complex_mul(b11r, b11i, x10r, x10i)
    z10r = bx10r + bx11r + y10r
    z10i = bx10i + bx11i + y10i

    bx12r, bx12i = complex_mul(b10r, b10i, x01r, x01i)
    bx13r, bx13i = complex_mul(b11r, b11i, x11r, x11i)
    z11r = bx12r + bx13r + y11r
    z11i = bx12i + bx13i + y11i

    return (
        c00r, c00i, c01r, c01i, c10r, c10i, c11r, c11i,
        z00r, z00i, z01r, z01i, z10r, z10i, z11r, z11i,
    )

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["L"],
)
@triton.jit
def mat2x2_pscan_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_a_batch: tl.constexpr, stride_a_time: tl.constexpr,
    stride_a_d1: tl.constexpr, stride_a_d2: tl.constexpr,
    stride_x_batch: tl.constexpr, stride_x_time: tl.constexpr,
    stride_x_d1: tl.constexpr, stride_x_d2: tl.constexpr,
    L, BLOCK_SIZE: tl.constexpr, REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_a_batch
    X_base = X_ptr + pid * stride_x_batch
    Y_base = Y_ptr + pid * stride_x_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    read_offs = L - 1 - offs if REVERSE else offs

    a00r = tl.load(A_base + read_offs * stride_a_time + 0 * stride_a_d1 + 0 * stride_a_d2 + 0, mask=mask, other=1.0)
    a00i = tl.load(A_base + read_offs * stride_a_time + 0 * stride_a_d1 + 0 * stride_a_d2 + 1, mask=mask, other=0.0)
    a01r = tl.load(A_base + read_offs * stride_a_time + 0 * stride_a_d1 + 1 * stride_a_d2 + 0, mask=mask, other=0.0)
    a01i = tl.load(A_base + read_offs * stride_a_time + 0 * stride_a_d1 + 1 * stride_a_d2 + 1, mask=mask, other=0.0)
    a10r = tl.load(A_base + read_offs * stride_a_time + 1 * stride_a_d1 + 0 * stride_a_d2 + 0, mask=mask, other=0.0)
    a10i = tl.load(A_base + read_offs * stride_a_time + 1 * stride_a_d1 + 0 * stride_a_d2 + 1, mask=mask, other=0.0)
    a11r = tl.load(A_base + read_offs * stride_a_time + 1 * stride_a_d1 + 1 * stride_a_d2 + 0, mask=mask, other=1.0)
    a11i = tl.load(A_base + read_offs * stride_a_time + 1 * stride_a_d1 + 1 * stride_a_d2 + 1, mask=mask, other=0.0)

    x00r = tl.load(X_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 0, mask=mask, other=0.0)
    x00i = tl.load(X_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 1, mask=mask, other=0.0)
    x01r = tl.load(X_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 0, mask=mask, other=0.0)
    x01i = tl.load(X_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 1, mask=mask, other=0.0)
    x10r = tl.load(X_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 0, mask=mask, other=0.0)
    x10i = tl.load(X_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 1, mask=mask, other=0.0)
    x11r = tl.load(X_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 0, mask=mask, other=0.0)
    x11i = tl.load(X_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 1, mask=mask, other=0.0)

    (_, _, _, _, _, _, _, _,
     y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i) = tl.associative_scan(
        (a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
         x00r, x00i, x01r, x01i, x10r, x10i, x11r, x11i),
        axis=0, combine_fn=mat2x2_scan_combine,
    )

    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 0, y00r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 1, y00i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 0, y01r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 1, y01i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 0, y10r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 1, y10i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 0, y11r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 1, y11i, mask=mask)

def run_mat2x2_pscan(A_real, X_real, L, reverse=False):
    Y_real = torch.empty_like(X_real)
    BLOCK_SIZE = max(16, next_power_of_2(L))
    num_batches = A_real.shape[0]

    mat2x2_pscan_kernel[(num_batches,)](
        A_real, X_real, Y_real,
        A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
        X_real.stride(0), X_real.stride(1), X_real.stride(2), X_real.stride(3),
        L, BLOCK_SIZE, reverse,
    )
    return Y_real

def expand_diag_to_matrix(A_diag, D):
    shape = A_diag.shape[:-1] + (D, D)
    A_mat = torch.zeros(shape, dtype=A_diag.dtype, device=A_diag.device)
    idx = torch.arange(D, device=A_diag.device)
    A_mat[..., idx, idx] = A_diag
    return A_mat

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, D1, D2_x = X.shape
        is_diag = (A.ndim == 4)
        ctx.is_diag = is_diag
        pad_d2 = (D2_x == 1)
        pad_d1 = (D1 == 1)
        ctx.pad_d2 = pad_d2
        ctx.pad_d1 = pad_d1

        X_in = X
        if pad_d1:
            X_in = F.pad(X_in, (0, 0, 0, 1), mode='constant', value=0.0)
        if pad_d2:
            X_in = F.pad(X_in, (0, 1), mode='constant', value=0.0)
            
        if is_diag:
            A_mat = expand_diag_to_matrix(A, D1)
        else:
            A_mat = A
        
        if pad_d1:
            A_padded = F.pad(A_mat, (0, 1, 0, 1), mode='constant', value=0.0)
            idx_pad = torch.zeros_like(A_padded)
            idx_pad[..., 1, 1] = 1.0
            A_mat = A_padded + idx_pad
            
        A_perm = A_mat.permute(0, 2, 1, 3, 4).contiguous()
        X_perm = X_in.permute(0, 2, 1, 3, 4).contiguous()

        A_flat = A_perm.reshape(B * C, L, A_mat.shape[-2], A_mat.shape[-1])
        X_flat = X_perm.reshape(B * C, L, X_in.shape[-2], X_in.shape[-1])

        A_real = torch.view_as_real(A_flat).contiguous()
        X_real = torch.view_as_real(X_flat).contiguous()
        
        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y_flat = torch.view_as_complex(Y_real.contiguous())
        Y_perm = Y_flat.reshape(B, C, L, X_in.shape[-2], X_in.shape[-1])
        Y_full = Y_perm.permute(0, 2, 1, 3, 4).contiguous()

        if pad_d1:
            Y_full = Y_full[..., :D1, :]
        if pad_d2:
            Y_full = Y_full[..., :D2_x]
            
        Y = Y_full
        ctx.save_for_backward(A_mat, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y = ctx.saved_tensors
        is_diag = ctx.is_diag
        pad_d2 = ctx.pad_d2
        pad_d1 = ctx.pad_d1

        B, L, C, D1_orig, D2_y = grad_Y.shape
        grad_Y_in = grad_Y
        Y_in = Y
        
        if pad_d1:
            grad_Y_in = F.pad(grad_Y_in, (0, 0, 0, 1), mode='constant', value=0.0)
            Y_in = F.pad(Y_in, (0, 0, 0, 1), mode='constant', value=0.0)
            
        if pad_d2:
            grad_Y_in = F.pad(grad_Y_in, (0, 1), mode='constant', value=0.0)
            Y_in = F.pad(Y_in, (0, 1), mode='constant', value=0.0)

        D1 = A_mat.shape[-2]
        D2_pad = grad_Y_in.shape[-1] 

        I = torch.eye(D1, dtype=A_mat.dtype, device=A_mat.device)
        I = I.reshape(1, 1, 1, D1, D1).expand(B, 1, C, D1, D1)

        A_shift = torch.cat([A_mat[:, 1:], I], dim=1)
        A_H = A_shift.conj().transpose(-1, -2)

        A_H_perm = A_H.permute(0, 2, 1, 3, 4).contiguous()
        grad_Y_perm = grad_Y_in.permute(0, 2, 1, 3, 4).contiguous()

        A_H_flat = A_H_perm.reshape(B * C, L, D1, D1)
        grad_Y_flat = grad_Y_perm.reshape(B * C, L, D1, D2_pad)

        A_H_real = torch.view_as_real(A_H_flat).contiguous()
        grad_Y_real = torch.view_as_real(grad_Y_flat).contiguous()
        grad_X_real = run_mat2x2_pscan(A_H_real, grad_Y_real, L, reverse=True)

        grad_X_flat = torch.view_as_complex(grad_X_real.contiguous())
        grad_X_perm = grad_X_flat.reshape(B, C, L, D1, D2_pad)
        grad_X_full = grad_X_perm.permute(0, 2, 1, 3, 4).contiguous()

        if pad_d1:
            grad_X_full = grad_X_full[..., :D1_orig, :]
        if pad_d2:
            grad_X = grad_X_full[..., :D2_y]
        else:
            grad_X = grad_X_full

        Y_prev = torch.cat([torch.zeros_like(Y_in[:, :1]), Y_in[:, :-1]], dim=1)
        grad_X_for_A = grad_X_flat.reshape(B, C, L, D1, D2_pad).permute(0, 2, 1, 3, 4).contiguous()
        grad_A_full = torch.einsum("blcij,blckj->blcik", grad_X_for_A, Y_prev.conj())

        if pad_d1:
             grad_A_full = grad_A_full[..., :D1_orig, :D1_orig]

        grad_A = grad_A_full.diagonal(dim1=-2, dim2=-1) if is_diag else grad_A_full
        return grad_A, grad_X

def pscan(A, X):
    is_diag = (A.ndim == X.ndim - 1) or (A.ndim == X.ndim and A.shape[-1] != X.shape[-1])
    if X.ndim == 4:
        X = X.unsqueeze(-1)
        squeeze_output = True
    else:
        squeeze_output = False

    A_mat = expand_diag_to_matrix(A, X.shape[-2]) if is_diag else A
    Y = _PScanFunction.apply(A_mat, X)
    return Y.squeeze(-1) if squeeze_output else Y
