import torch
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
    a00r,
    a00i,
    a01r,
    a01i,
    a10r,
    a10i,
    a11r,
    a11i,
    x00r,
    x00i,
    x01r,
    x01i,
    x10r,
    x10i,
    x11r,
    x11i,
    b00r,
    b00i,
    b01r,
    b01i,
    b10r,
    b10i,
    b11r,
    b11i,
    y00r,
    y00i,
    y01r,
    y01i,
    y10r,
    y10i,
    y11r,
    y11i,
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
        c00r,
        c00i,
        c01r,
        c01i,
        c10r,
        c10i,
        c11r,
        c11i,
        z00r,
        z00i,
        z01r,
        z01i,
        z10r,
        z10i,
        z11r,
        z11i,
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
    A_ptr,
    X_ptr,
    Y_ptr,
    stride_a_batch,
    stride_a_time,
    stride_a_d1,
    stride_a_d2,
    stride_x_batch,
    stride_x_time,
    stride_x_d1,
    stride_x_d2,
    L,
    BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    a_base = A_ptr + pid * stride_a_batch
    x_base = X_ptr + pid * stride_x_batch
    y_base = Y_ptr + pid * stride_x_batch
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    read_offs = L - 1 - offs if REVERSE else offs
    a00r = tl.load(a_base + read_offs * stride_a_time + 0, mask=mask, other=1.0)
    a00i = tl.load(a_base + read_offs * stride_a_time + 1, mask=mask, other=0.0)
    a01r = tl.load(a_base + read_offs * stride_a_time + stride_a_d2 + 0, mask=mask, other=0.0)
    a01i = tl.load(a_base + read_offs * stride_a_time + stride_a_d2 + 1, mask=mask, other=0.0)
    a10r = tl.load(a_base + read_offs * stride_a_time + stride_a_d1 + 0, mask=mask, other=0.0)
    a10i = tl.load(a_base + read_offs * stride_a_time + stride_a_d1 + 1, mask=mask, other=0.0)
    a11r = tl.load(a_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 0, mask=mask, other=1.0)
    a11i = tl.load(a_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 1, mask=mask, other=0.0)
    x00r = tl.load(x_base + read_offs * stride_x_time + 0, mask=mask, other=0.0)
    x00i = tl.load(x_base + read_offs * stride_x_time + 1, mask=mask, other=0.0)
    x01r = tl.load(x_base + read_offs * stride_x_time + stride_x_d2 + 0, mask=mask, other=0.0)
    x01i = tl.load(x_base + read_offs * stride_x_time + stride_x_d2 + 1, mask=mask, other=0.0)
    x10r = tl.load(x_base + read_offs * stride_x_time + stride_x_d1 + 0, mask=mask, other=0.0)
    x10i = tl.load(x_base + read_offs * stride_x_time + stride_x_d1 + 1, mask=mask, other=0.0)
    x11r = tl.load(x_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0, mask=mask, other=0.0)
    x11i = tl.load(x_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1, mask=mask, other=0.0)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        y00r,
        y00i,
        y01r,
        y01i,
        y10r,
        y10i,
        y11r,
        y11i,
    ) = tl.associative_scan(
        (
            a00r,
            a00i,
            a01r,
            a01i,
            a10r,
            a10i,
            a11r,
            a11i,
            x00r,
            x00i,
            x01r,
            x01i,
            x10r,
            x10i,
            x11r,
            x11i,
        ),
        axis=0,
        combine_fn=mat2x2_scan_combine,
    )
    tl.store(y_base + read_offs * stride_x_time + 0, y00r, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + 1, y00i, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d2 + 0, y01r, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d2 + 1, y01i, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d1 + 0, y10r, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d1 + 1, y10i, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0, y11r, mask=mask)
    tl.store(y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1, y11i, mask=mask)


def run_mat2x2_pscan(a_real, x_real, length, reverse):
    y_real = torch.empty_like(x_real)
    block_size = max(16, next_power_of_2(int(length)))
    mat2x2_pscan_kernel[(a_real.shape[0],)](
        a_real,
        x_real,
        y_real,
        a_real.stride(0),
        a_real.stride(1),
        a_real.stride(2),
        a_real.stride(3),
        x_real.stride(0),
        x_real.stride(1),
        x_real.stride(2),
        x_real.stride(3),
        int(length),
        block_size,
        reverse,
    )
    return y_real


def _expand_diag_to_matrix(a_diag):
    n, length, channels, _ = a_diag.shape
    a_diag = a_diag[..., 0]
    a_mat = torch.zeros((n, length, channels, 2, 2), dtype=a_diag.dtype, device=a_diag.device)
    a_mat[..., 0, 0] = a_diag
    a_mat[..., 1, 1] = 1.0
    return a_mat


class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_diag, x_diag):
        n, length, channels, _ = x_diag.shape
        a_mat = _expand_diag_to_matrix(a_diag)
        x_mat = torch.zeros((n, length, channels, 2, 2), dtype=x_diag.dtype, device=x_diag.device)
        x_mat[..., 0, 0] = x_diag[..., 0]
        a_flat = a_mat.permute(0, 2, 1, 3, 4).reshape(n * channels, length, 2, 2)
        x_flat = x_mat.permute(0, 2, 1, 3, 4).reshape(n * channels, length, 2, 2)
        a_real = torch.view_as_real(a_flat).contiguous()
        x_real = torch.view_as_real(x_flat).contiguous()
        y_real = run_mat2x2_pscan(a_real, x_real, length, reverse=False)
        y_flat = torch.view_as_complex(y_real.contiguous())
        y_mat = y_flat.reshape(n, channels, length, 2, 2).permute(0, 2, 1, 3, 4).contiguous()
        y_out = y_mat[..., 0, 0].unsqueeze(-1)
        ctx.save_for_backward(a_diag, y_out)
        return y_out

    @staticmethod
    def backward(ctx, grad_y):
        a_diag, y_out = ctx.saved_tensors
        n, length, channels, _ = y_out.shape
        a_mat = _expand_diag_to_matrix(a_diag)
        i_mat = torch.eye(2, dtype=a_mat.dtype, device=a_mat.device).view(1, 1, 1, 2, 2).expand(n, 1, channels, 2, 2)
        a_shift = torch.cat([a_mat[:, 1:], i_mat], dim=1)
        a_h = a_shift.conj().transpose(-1, -2)
        grad_mat = torch.zeros((n, length, channels, 2, 2), dtype=grad_y.dtype, device=grad_y.device)
        grad_mat[..., 0, 0] = grad_y[..., 0]
        a_h_flat = a_h.permute(0, 2, 1, 3, 4).reshape(n * channels, length, 2, 2)
        grad_flat = grad_mat.permute(0, 2, 1, 3, 4).reshape(n * channels, length, 2, 2)
        a_h_real = torch.view_as_real(a_h_flat.resolve_conj()).contiguous()
        grad_real = torch.view_as_real(grad_flat).contiguous()
        grad_x_real = run_mat2x2_pscan(a_h_real, grad_real, length, reverse=True)
        grad_x_flat = torch.view_as_complex(grad_x_real.contiguous())
        grad_x_mat = grad_x_flat.reshape(n, channels, length, 2, 2).permute(0, 2, 1, 3, 4).contiguous()
        grad_x = grad_x_mat[..., 0, 0].unsqueeze(-1)
        y_prev = torch.cat([torch.zeros_like(y_out[:, :1]), y_out[:, :-1]], dim=1)
        grad_a = grad_x[..., 0] * y_prev[..., 0].conj()
        grad_a = grad_a.unsqueeze(-1)
        return grad_a, grad_x


def pscan(a_diag, x_diag):
    return _PScanFunction.apply(a_diag, x_diag)
