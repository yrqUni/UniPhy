import torch
import torch.nn as nn
import triton
import triton.language as tl


def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def get_autotune_configs():
    return [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]


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


@triton.autotune(configs=get_autotune_configs(), key=["L"])
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


@triton.autotune(configs=get_autotune_configs(), key=["L"])
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


def complex_to_real(z):
    return torch.stack([z.real, z.imag], dim=-1)


def real_to_complex(r):
    return torch.complex(r[..., 0], r[..., 1])


class _PScanDiagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        A_orig_shape = A.shape
        X_orig_shape = X.shape

        is_diag_a = A.ndim == X.ndim - 1

        if is_diag_a:
            D = A.shape[-1]
            A_expanded = torch.zeros(
                *A.shape, D, dtype=A.dtype, device=A.device
            )
            idx = torch.arange(D, device=A.device)
            A_expanded[..., idx, idx] = A
            A_mat = A_expanded
        else:
            A_mat = A

        B_flat = A_mat.shape[0] * A_mat.shape[2]
        L = A_mat.shape[1]

        A_real = complex_to_real(
            A_mat.permute(0, 2, 1, 3, 4).reshape(B_flat, L, 2, 2)
        ).contiguous()
        X_real = complex_to_real(
            X.permute(0, 2, 1, 3, 4).reshape(B_flat, L, 2, 2)
        ).contiguous()

        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y = real_to_complex(Y_real).reshape(
            X.shape[0], X.shape[2], L, 2, 2
        ).permute(0, 2, 1, 3, 4)

        ctx.save_for_backward(A_mat, Y)
        ctx.is_diag_a = is_diag_a
        ctx.A_orig_shape = A_orig_shape

        return Y.contiguous()

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y = ctx.saved_tensors
        is_diag_a = ctx.is_diag_a
        A_orig_shape = ctx.A_orig_shape

        B, L, C, D, _ = grad_Y.shape

        Y_shifted = torch.zeros_like(Y)
        Y_shifted[:, 1:] = Y[:, :-1]

        grad_A_mat = torch.einsum("blcij,blckj->blcik", grad_Y, Y_shifted.conj())

        A_conj = A_mat.conj()
        B_flat = B * C
        L = grad_Y.shape[1]

        grad_Y_real = complex_to_real(
            grad_Y.permute(0, 2, 1, 3, 4).reshape(B_flat, L, 2, 2)
        ).contiguous()
        A_conj_real = complex_to_real(
            A_conj.permute(0, 2, 1, 3, 4).reshape(B_flat, L, 2, 2)
        ).contiguous()

        grad_X_real = run_mat2x2_pscan(A_conj_real, grad_Y_real, L, reverse=True)

        grad_X = real_to_complex(grad_X_real).reshape(B, C, L, 2, 2).permute(0, 2, 1, 3, 4)

        if is_diag_a:
            idx = torch.arange(D, device=grad_A_mat.device)
            grad_A = grad_A_mat[..., idx, idx]
        else:
            grad_A = grad_A_mat

        return grad_A.contiguous(), grad_X.contiguous()


class _PScanMatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, D1, D2 = A.shape
        B_flat = B * C

        A_real = complex_to_real(
            A.permute(0, 2, 1, 3, 4).reshape(B_flat, L, D1, D2)
        ).contiguous()
        X_real = complex_to_real(
            X.permute(0, 2, 1, 3, 4).reshape(B_flat, L, D1, D2)
        ).contiguous()

        Y_real = run_mat2x2_pscan(A_real, X_real, L, reverse=False)

        Y = real_to_complex(Y_real).reshape(B, C, L, D1, D2).permute(0, 2, 1, 3, 4)

        ctx.save_for_backward(A, Y)

        return Y.contiguous()

    @staticmethod
    def backward(ctx, grad_Y):
        A, Y = ctx.saved_tensors
        B, L, C, D1, D2 = grad_Y.shape

        Y_shifted = torch.zeros_like(Y)
        Y_shifted[:, 1:] = Y[:, :-1]

        grad_A = torch.einsum("blcij,blckj->blcik", grad_Y, Y_shifted.conj())

        A_conj = A.conj()
        B_flat = B * C

        grad_Y_real = complex_to_real(
            grad_Y.permute(0, 2, 1, 3, 4).reshape(B_flat, L, D1, D2)
        ).contiguous()
        A_conj_real = complex_to_real(
            A_conj.permute(0, 2, 1, 3, 4).reshape(B_flat, L, D1, D2)
        ).contiguous()

        grad_X_real = run_mat2x2_pscan(A_conj_real, grad_Y_real, L, reverse=True)

        grad_X = real_to_complex(grad_X_real).reshape(B, C, L, D1, D2).permute(0, 2, 1, 3, 4)

        return grad_A.contiguous(), grad_X.contiguous()


def pscan(A, X, mode="auto"):
    is_diag = A.ndim == X.ndim - 1

    if mode == "diag" or (mode == "auto" and is_diag):
        return _PScanDiagFunction.apply(A, X)
    else:
        return _PScanMatFunction.apply(A, X)


class PScanTriton(nn.Module):
    def __init__(self, mode="auto"):
        super().__init__()
        self.mode = mode

    def forward(self, A, X):
        return pscan(A, X, mode=self.mode)
    