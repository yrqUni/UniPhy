import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def next_power_of_2(n):
    if n <= 0:
        return 1
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
    c00r, c00i = c00r + t1r, c00i + t1i

    c01r, c01i = complex_mul(b00r, b00i, a01r, a01i)
    t2r, t2i = complex_mul(b01r, b01i, a11r, a11i)
    c01r, c01i = c01r + t2r, c01i + t2i

    c10r, c10i = complex_mul(b10r, b10i, a00r, a00i)
    t3r, t3i = complex_mul(b11r, b11i, a10r, a10i)
    c10r, c10i = c10r + t3r, c10i + t3i

    c11r, c11i = complex_mul(b10r, b10i, a01r, a01i)
    t4r, t4i = complex_mul(b11r, b11i, a11r, a11i)
    c11r, c11i = c11r + t4r, c11i + t4i

    bx00r, bx00i = complex_mul(b00r, b00i, x00r, x00i)
    bx01r, bx01i = complex_mul(b01r, b01i, x10r, x10i)
    z00r, z00i = bx00r + bx01r + y00r, bx00i + bx01i + y00i

    bx02r, bx02i = complex_mul(b00r, b00i, x01r, x01i)
    bx03r, bx03i = complex_mul(b01r, b01i, x11r, x11i)
    z01r, z01i = bx02r + bx03r + y01r, bx02i + bx03i + y01i

    bx10r, bx10i = complex_mul(b10r, b10i, x00r, x00i)
    bx11r, bx11i = complex_mul(b11r, b11i, x10r, x10i)
    z10r, z10i = bx10r + bx11r + y10r, bx10i + bx11i + y10i

    bx12r, bx12i = complex_mul(b10r, b10i, x01r, x01i)
    bx13r, bx13i = complex_mul(b11r, b11i, x11r, x11i)
    z11r, z11i = bx12r + bx13r + y11r, bx12i + bx13i + y11i

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
    stride_a_batch, stride_a_time, stride_a_d1, stride_a_d2,
    stride_x_batch, stride_x_time, stride_x_d1, stride_x_d2,
    L, BLOCK_SIZE: tl.constexpr, REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_a_batch
    X_base = X_ptr + pid * stride_x_batch
    Y_base = Y_ptr + pid * stride_x_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    read_offs = L - 1 - offs if REVERSE else offs

    a00r = tl.load(A_base + read_offs * stride_a_time + 0, mask=mask, other=1.0)
    a00i = tl.load(A_base + read_offs * stride_a_time + 1, mask=mask, other=0.0)
    a01r = tl.load(A_base + read_offs * stride_a_time + stride_a_d2 + 0, mask=mask, other=0.0)
    a01i = tl.load(A_base + read_offs * stride_a_time + stride_a_d2 + 1, mask=mask, other=0.0)
    a10r = tl.load(A_base + read_offs * stride_a_time + stride_a_d1 + 0, mask=mask, other=0.0)
    a10i = tl.load(A_base + read_offs * stride_a_time + stride_a_d1 + 1, mask=mask, other=0.0)
    a11r = tl.load(A_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 0, mask=mask, other=1.0)
    a11i = tl.load(A_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 1, mask=mask, other=0.0)

    x00r = tl.load(X_base + read_offs * stride_x_time + 0, mask=mask, other=0.0)
    x00i = tl.load(X_base + read_offs * stride_x_time + 1, mask=mask, other=0.0)
    x01r = tl.load(X_base + read_offs * stride_x_time + stride_x_d2 + 0, mask=mask, other=0.0)
    x01i = tl.load(X_base + read_offs * stride_x_time + stride_x_d2 + 1, mask=mask, other=0.0)
    x10r = tl.load(X_base + read_offs * stride_x_time + stride_x_d1 + 0, mask=mask, other=0.0)
    x10i = tl.load(X_base + read_offs * stride_x_time + stride_x_d1 + 1, mask=mask, other=0.0)
    x11r = tl.load(X_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0, mask=mask, other=0.0)
    x11i = tl.load(X_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1, mask=mask, other=0.0)

    (_, _, _, _, _, _, _, _,
        y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i) = tl.associative_scan(
        (a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
            x00r, x00i, x01r, x01i, x10r, x10i, x11r, x11i),
        axis=0, combine_fn=mat2x2_scan_combine,
    )

    tl.store(Y_base + read_offs * stride_x_time + 0, y00r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1, y00i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d2 + 0, y01r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d2 + 1, y01i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + 0, y10r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + 1, y10i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0, y11r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1, y11i, mask=mask)


def run_mat2x2_pscan_torch(A_complex, X_complex, reverse=False):
    B, L, D1, D2 = X_complex.shape
    Y = torch.zeros_like(X_complex)
    indices = range(L - 1, -1, -1) if reverse else range(L)
    acc = torch.zeros(B, D1, D2, dtype=X_complex.dtype, device=X_complex.device)

    for i in indices:
        acc = torch.bmm(A_complex[:, i], acc) + X_complex[:, i]
        Y[:, i] = acc

    return Y


def run_mat2x2_pscan(A_real, X_real, L, reverse=False):
    Y_real = torch.empty_like(X_real)
    BLOCK_SIZE = max(16, next_power_of_2(L))
    mat2x2_pscan_kernel[(A_real.shape[0],)](
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
    def forward(ctx, A, X, is_diag):
        B, L, C, D1, D2_x = X.shape
        pad_d2 = (D2_x == 1)
        pad_d1 = (D1 == 1)

        ctx.is_diag = is_diag
        ctx.pad_d2 = pad_d2
        ctx.pad_d1 = pad_d1
        ctx.D1_orig = D1
        ctx.D2_orig = D2_x

        X_in = X
        if pad_d1:
            X_in = F.pad(X_in, (0, 0, 0, 1), mode="constant", value=0.0)
        if pad_d2:
            X_in = F.pad(X_in, (0, 1), mode="constant", value=0.0)

        A_mat = expand_diag_to_matrix(A, D1) if is_diag else A

        if pad_d1:
            A_padded = F.pad(A_mat, (0, 1, 0, 1), mode="constant", value=0.0)
            A_padded[..., -1, -1] = 1.0
            A_mat = A_padded

        D1_pad, D2_pad = X_in.shape[-2], X_in.shape[-1]

        A_flat = A_mat.permute(0, 2, 1, 3, 4).reshape(B * C, L, A_mat.shape[-2], A_mat.shape[-1])
        X_flat = X_in.permute(0, 2, 1, 3, 4).reshape(B * C, L, D1_pad, D2_pad)

        A_real = torch.view_as_real(A_flat).contiguous()
        X_real = torch.view_as_real(X_flat).contiguous()

        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y_flat = torch.view_as_complex(Y_real.contiguous())
        Y_full = Y_flat.reshape(B, C, L, D1_pad, D2_pad).permute(0, 2, 1, 3, 4).contiguous()

        if pad_d1:
            Y_full = Y_full[..., :D1, :]
        if pad_d2:
            Y_full = Y_full[..., :D2_x]

        ctx.save_for_backward(A_mat, Y_flat.reshape(B, C, L, D1_pad, D2_pad).permute(0, 2, 1, 3, 4).contiguous())
        ctx.B, ctx.L, ctx.C, ctx.D1_pad, ctx.D2_pad = B, L, C, D1_pad, D2_pad

        return Y_full

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y_saved = ctx.saved_tensors
        B, L, C = ctx.B, ctx.L, ctx.C
        D1_pad, D2_pad = ctx.D1_pad, ctx.D2_pad
        D1_orig, D2_orig = ctx.D1_orig, ctx.D2_orig

        grad_Y_in = grad_Y
        if ctx.pad_d1:
            grad_Y_in = F.pad(grad_Y_in, (0, 0, 0, 1), mode="constant", value=0.0)
        if ctx.pad_d2:
            grad_Y_in = F.pad(grad_Y_in, (0, 1), mode="constant", value=0.0)

        D1 = A_mat.shape[-2]
        I = torch.eye(D1, dtype=A_mat.dtype, device=A_mat.device).reshape(1, 1, 1, D1, D1).expand(B, 1, C, D1, D1)
        A_shift = torch.cat([A_mat[:, 1:], I], dim=1)
        A_H = A_shift.conj().transpose(-1, -2)

        A_H_flat = A_H.permute(0, 2, 1, 3, 4).reshape(B * C, L, D1, D1)
        grad_Y_flat = grad_Y_in.permute(0, 2, 1, 3, 4).reshape(B * C, L, D1_pad, D2_pad)

        A_H_real = torch.view_as_real(A_H_flat.resolve_conj()).contiguous()
        grad_Y_real = torch.view_as_real(grad_Y_flat).contiguous()

        grad_X_real = run_mat2x2_pscan(A_H_real, grad_Y_real, L, reverse=True)

        grad_X_flat = torch.view_as_complex(grad_X_real.contiguous())
        grad_X_full = grad_X_flat.reshape(B, C, L, D1_pad, D2_pad).permute(0, 2, 1, 3, 4).contiguous()

        if ctx.pad_d1:
            grad_X_full = grad_X_full[..., :D1_orig, :]
        grad_X = grad_X_full[..., :D2_orig] if ctx.pad_d2 else grad_X_full

        Y_prev = torch.cat([torch.zeros_like(Y_saved[:, :1]), Y_saved[:, :-1]], dim=1)
        grad_X_for_A = grad_X_flat.reshape(B, C, L, D1_pad, D2_pad).permute(0, 2, 1, 3, 4).contiguous()
        grad_A_full = torch.einsum("blcij,blckj->blcik", grad_X_for_A, Y_prev.conj())

        if ctx.pad_d1:
            grad_A_full = grad_A_full[..., :D1_orig, :D1_orig]

        grad_A = grad_A_full.diagonal(dim1=-2, dim2=-1) if ctx.is_diag else grad_A_full
        return grad_A, grad_X, None


def pscan(A, X):
    squeeze_output = X.ndim == 4

    if squeeze_output:
        X = X.unsqueeze(-1)

    is_diag = (A.ndim == 4)

    Y = _PScanFunction.apply(A, X, is_diag)

    return Y.squeeze(-1) if squeeze_output else Y
