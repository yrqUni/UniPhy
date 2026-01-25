import torch
import torch.nn as nn
import triton
import triton.language as tl


def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def diag_scan_combine(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar, new_ai = complex_mul(br, bi, ar, ai)
    bx_r, bx_i = complex_mul(br, bi, xr, xi)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi


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
def diag_pscan_kernel(
    A_ptr,
    X_ptr,
    Y_ptr,
    stride_batch: tl.constexpr,
    stride_time: tl.constexpr,
    L,
    BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    if REVERSE:
        read_offs = L - 1 - offs
    else:
        read_offs = offs

    a_r = tl.load(A_base + read_offs * stride_time + 0, mask=mask, other=1.0)
    a_i = tl.load(A_base + read_offs * stride_time + 1, mask=mask, other=0.0)
    x_r = tl.load(X_base + read_offs * stride_time + 0, mask=mask, other=0.0)
    x_i = tl.load(X_base + read_offs * stride_time + 1, mask=mask, other=0.0)

    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan(
        (a_r, a_i, x_r, x_i), axis=0, combine_fn=diag_scan_combine
    )

    tl.store(Y_base + read_offs * stride_time + 0, acc_xr, mask=mask)
    tl.store(Y_base + read_offs * stride_time + 1, acc_xi, mask=mask)


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
    A_ptr,
    X_ptr,
    Y_ptr,
    stride_a_batch: tl.constexpr,
    stride_a_time: tl.constexpr,
    stride_a_d1: tl.constexpr,
    stride_a_d2: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_time: tl.constexpr,
    stride_x_d1: tl.constexpr,
    stride_x_d2: tl.constexpr,
    L,
    BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    A_base = A_ptr + pid * stride_a_batch
    X_base = X_ptr + pid * stride_x_batch
    Y_base = Y_ptr + pid * stride_x_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    if REVERSE:
        read_offs = L - 1 - offs
    else:
        read_offs = offs

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

    (
        _, _, _, _, _, _, _, _,
        y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i,
    ) = tl.associative_scan(
        (
            a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
            x00r, x00i, x01r, x01i, x10r, x10i, x11r, x11i,
        ),
        axis=0,
        combine_fn=mat2x2_scan_combine,
    )

    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 0, y00r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 0 * stride_x_d2 + 1, y00i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 0, y01r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 0 * stride_x_d1 + 1 * stride_x_d2 + 1, y01i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 0, y10r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 0 * stride_x_d2 + 1, y10i, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 0, y11r, mask=mask)
    tl.store(Y_base + read_offs * stride_x_time + 1 * stride_x_d1 + 1 * stride_x_d2 + 1, y11i, mask=mask)


def run_diag_pscan(A_real, X_real, L, reverse=False):
    Y_real = torch.empty_like(X_real)
    BLOCK_SIZE = max(16, next_power_of_2(L))
    num_batches = A_real.shape[0]

    diag_pscan_kernel[(num_batches,)](
        A_real,
        X_real,
        Y_real,
        A_real.stride(0),
        A_real.stride(1),
        L,
        BLOCK_SIZE,
        reverse,
    )
    return Y_real


def run_mat2x2_pscan(A_real, X_real, L, reverse=False):
    Y_real = torch.empty_like(X_real)
    BLOCK_SIZE = max(16, next_power_of_2(L))
    num_batches = A_real.shape[0]

    mat2x2_pscan_kernel[(num_batches,)](
        A_real,
        X_real,
        Y_real,
        A_real.stride(0),
        A_real.stride(1),
        A_real.stride(2),
        A_real.stride(3),
        X_real.stride(0),
        X_real.stride(1),
        X_real.stride(2),
        X_real.stride(3),
        L,
        BLOCK_SIZE,
        reverse,
    )
    return Y_real


def expand_diag_to_matrix(A_diag, D):
    B, L, C, _ = A_diag.shape
    A_mat = torch.zeros(B, L, C, D, D, dtype=A_diag.dtype, device=A_diag.device)
    idx = torch.arange(D, device=A_diag.device)
    A_mat[..., idx, idx] = A_diag
    return A_mat


class _PScanDiagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, D, _ = X.shape
        ctx.shape_A_orig = A.shape
        ctx.shape_X_orig = X.shape

        is_diag = (A.ndim == 4)
        ctx.is_diag = is_diag

        if is_diag:
            A_mat = expand_diag_to_matrix(A, D)
        else:
            A_mat = A

        A_work = A_mat.contiguous()
        X_work = X.contiguous()

        A_perm = A_work.permute(0, 2, 1, 3, 4).contiguous()
        X_perm = X_work.permute(0, 2, 1, 3, 4).contiguous()

        A_flat = A_perm.reshape(B * C, L, D, D)
        X_flat = X_perm.reshape(B * C, L, D, D)

        A_real = torch.view_as_real(A_flat).contiguous()
        X_real = torch.view_as_real(X_flat).contiguous()

        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y_flat = torch.view_as_complex(Y_real)
        Y_perm = Y_flat.reshape(B, C, L, D, D)
        Y = Y_perm.permute(0, 2, 1, 3, 4).contiguous()

        ctx.save_for_backward(A_mat, Y)

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y = ctx.saved_tensors
        is_diag = ctx.is_diag

        B, L, C, D, _ = grad_Y.shape

        A_conj = A_mat.conj()

        grad_Y_perm = grad_Y.permute(0, 2, 1, 3, 4).contiguous()
        A_conj_perm = A_conj.permute(0, 2, 1, 3, 4).contiguous()

        grad_Y_flat = grad_Y_perm.reshape(B * C, L, D, D)
        A_conj_flat = A_conj_perm.reshape(B * C, L, D, D)

        grad_Y_real = torch.view_as_real(grad_Y_flat).contiguous()
        A_conj_real = torch.view_as_real(A_conj_flat).contiguous()

        dX_real = run_mat2x2_pscan(A_conj_real, grad_Y_real, L, reverse=True)

        dX_flat = torch.view_as_complex(dX_real)
        dX_perm = dX_flat.reshape(B, C, L, D, D)
        dX = dX_perm.permute(0, 2, 1, 3, 4).contiguous()

        Y_prev = torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
        dA_full = torch.einsum("blcij,blckj->blcik", grad_Y, Y_prev.conj())

        if is_diag:
            dA = dA_full.diagonal(dim1=-2, dim2=-1)
        else:
            dA = dA_full

        return dA, dX


class _PScanMatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, D1, D2 = X.shape
        ctx.shape_A_orig = A.shape
        ctx.shape_X_orig = X.shape

        is_diag = (A.ndim == 4)
        ctx.is_diag = is_diag

        if is_diag:
            A_mat = expand_diag_to_matrix(A, D1)
        else:
            A_mat = A

        A_work = A_mat.contiguous()
        X_work = X.contiguous()

        A_perm = A_work.permute(0, 2, 1, 3, 4).contiguous()
        X_perm = X_work.permute(0, 2, 1, 3, 4).contiguous()

        A_flat = A_perm.reshape(B * C, L, D1, D2)
        X_flat = X_perm.reshape(B * C, L, D1, D2)

        A_real = torch.view_as_real(A_flat).contiguous()
        X_real = torch.view_as_real(X_flat).contiguous()

        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y_flat = torch.view_as_complex(Y_real)
        Y_perm = Y_flat.reshape(B, C, L, D1, D2)
        Y = Y_perm.permute(0, 2, 1, 3, 4).contiguous()

        ctx.save_for_backward(A_mat, Y)

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y = ctx.saved_tensors
        is_diag = ctx.is_diag

        B, L, C, D1, D2 = grad_Y.shape

        A_conj = A_mat.conj()

        grad_Y_perm = grad_Y.permute(0, 2, 1, 3, 4).contiguous()
        A_conj_perm = A_conj.permute(0, 2, 1, 3, 4).contiguous()

        grad_Y_flat = grad_Y_perm.reshape(B * C, L, D1, D2)
        A_conj_flat = A_conj_perm.reshape(B * C, L, D1, D2)

        grad_Y_real = torch.view_as_real(grad_Y_flat).contiguous()
        A_conj_real = torch.view_as_real(A_conj_flat).contiguous()

        dX_real = run_mat2x2_pscan(A_conj_real, grad_Y_real, L, reverse=True)

        dX_flat = torch.view_as_complex(dX_real)
        dX_perm = dX_flat.reshape(B, C, L, D1, D2)
        dX = dX_perm.permute(0, 2, 1, 3, 4).contiguous()

        Y_prev = torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
        dA_full = torch.einsum("blcij,blckj->blcik", grad_Y, Y_prev.conj())

        if is_diag:
            dA = dA_full.diagonal(dim1=-2, dim2=-1)
        else:
            dA = dA_full

        return dA, dX


def pscan(A, X, mode="auto"):
    is_diag = (A.ndim == X.ndim - 1)

    if mode == "diag" or (mode == "auto" and is_diag):
        return _PScanDiagFunction.apply(A, X)
    else:
        return _PScanMatFunction.apply(A, X)
