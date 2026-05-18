import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


if triton is not None:

    @triton.jit
    def _complex_mul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br

    @triton.jit
    def _mat2x2_scan_combine(
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
        c00r, c00i = _complex_mul(b00r, b00i, a00r, a00i)
        t1r, t1i = _complex_mul(b01r, b01i, a10r, a10i)
        c00r, c00i = c00r + t1r, c00i + t1i
        c01r, c01i = _complex_mul(b00r, b00i, a01r, a01i)
        t2r, t2i = _complex_mul(b01r, b01i, a11r, a11i)
        c01r, c01i = c01r + t2r, c01i + t2i
        c10r, c10i = _complex_mul(b10r, b10i, a00r, a00i)
        t3r, t3i = _complex_mul(b11r, b11i, a10r, a10i)
        c10r, c10i = c10r + t3r, c10i + t3i
        c11r, c11i = _complex_mul(b10r, b10i, a01r, a01i)
        t4r, t4i = _complex_mul(b11r, b11i, a11r, a11i)
        c11r, c11i = c11r + t4r, c11i + t4i
        bx00r, bx00i = _complex_mul(b00r, b00i, x00r, x00i)
        bx01r, bx01i = _complex_mul(b01r, b01i, x10r, x10i)
        z00r, z00i = bx00r + bx01r + y00r, bx00i + bx01i + y00i
        bx02r, bx02i = _complex_mul(b00r, b00i, x01r, x01i)
        bx03r, bx03i = _complex_mul(b01r, b01i, x11r, x11i)
        z01r, z01i = bx02r + bx03r + y01r, bx02i + bx03i + y01i
        bx10r, bx10i = _complex_mul(b10r, b10i, x00r, x00i)
        bx11r, bx11i = _complex_mul(b11r, b11i, x10r, x10i)
        z10r, z10i = bx10r + bx11r + y10r, bx10i + bx11i + y10i
        bx12r, bx12i = _complex_mul(b10r, b10i, x01r, x01i)
        bx13r, bx13i = _complex_mul(b11r, b11i, x11r, x11i)
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
    def _mat2x2_pscan_kernel(
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
        A_base = A_ptr + pid * stride_a_batch
        X_base = X_ptr + pid * stride_x_batch
        Y_base = Y_ptr + pid * stride_x_batch
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < L
        read_offs = L - 1 - offs if REVERSE else offs
        a00r = tl.load(A_base + read_offs * stride_a_time + 0, mask=mask, other=1.0)
        a00i = tl.load(A_base + read_offs * stride_a_time + 1, mask=mask, other=0.0)
        a01r = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d2 + 0,
            mask=mask,
            other=0.0,
        )
        a01i = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d2 + 1,
            mask=mask,
            other=0.0,
        )
        a10r = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d1 + 0,
            mask=mask,
            other=0.0,
        )
        a10i = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d1 + 1,
            mask=mask,
            other=0.0,
        )
        a11r = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 0,
            mask=mask,
            other=1.0,
        )
        a11i = tl.load(
            A_base + read_offs * stride_a_time + stride_a_d1 + stride_a_d2 + 1,
            mask=mask,
            other=0.0,
        )
        x00r = tl.load(X_base + read_offs * stride_x_time + 0, mask=mask, other=0.0)
        x00i = tl.load(X_base + read_offs * stride_x_time + 1, mask=mask, other=0.0)
        x01r = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d2 + 0,
            mask=mask,
            other=0.0,
        )
        x01i = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d2 + 1,
            mask=mask,
            other=0.0,
        )
        x10r = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d1 + 0,
            mask=mask,
            other=0.0,
        )
        x10i = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d1 + 1,
            mask=mask,
            other=0.0,
        )
        x11r = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0,
            mask=mask,
            other=0.0,
        )
        x11i = tl.load(
            X_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1,
            mask=mask,
            other=0.0,
        )
        (_, _, _, _, _, _, _, _, y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i) = (
            tl.associative_scan(
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
                combine_fn=_mat2x2_scan_combine,
            )
        )
        tl.store(Y_base + read_offs * stride_x_time + 0, y00r, mask=mask)
        tl.store(Y_base + read_offs * stride_x_time + 1, y00i, mask=mask)
        tl.store(Y_base + read_offs * stride_x_time + stride_x_d2 + 0, y01r, mask=mask)
        tl.store(Y_base + read_offs * stride_x_time + stride_x_d2 + 1, y01i, mask=mask)
        tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + 0, y10r, mask=mask)
        tl.store(Y_base + read_offs * stride_x_time + stride_x_d1 + 1, y10i, mask=mask)
        tl.store(
            Y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 0,
            y11r,
            mask=mask,
        )
        tl.store(
            Y_base + read_offs * stride_x_time + stride_x_d1 + stride_x_d2 + 1,
            y11i,
            mask=mask,
        )


def _validate_pscan_inputs(a, x):
    squeeze_output = x.ndim == 4
    if squeeze_output:
        x = x.unsqueeze(-1)
    if x.ndim != 5:
        raise ValueError(f"unsupported_x_rank={x.ndim}")
    if a.ndim not in {4, 5}:
        raise ValueError(f"unsupported_a_rank={a.ndim}")
    if a.shape[:3] != x.shape[:3]:
        raise ValueError(f"shape_mismatch_a={a.shape} x={x.shape}")
    is_diag = a.ndim == 4
    if is_diag:
        if a.shape[-1] != x.shape[-2]:
            raise ValueError(f"diag_state_mismatch_a={a.shape} x={x.shape}")
    else:
        if a.shape[-1] != a.shape[-2]:
            raise ValueError(f"nonsquare_transition={a.shape}")
        if a.shape[-1] != x.shape[-2]:
            raise ValueError(f"matrix_state_mismatch_a={a.shape} x={x.shape}")
    return a, x, is_diag, squeeze_output


def _interleave_tree(left, right):
    return torch.stack((left, right), dim=2).flatten(1, 2)


def _combine_diag(a_left, x_left, a_right, x_right):
    return (
        a_right * a_left,
        a_right.unsqueeze(-1) * x_left + x_right,
    )


def _combine_mat(a_left, x_left, a_right, x_right):
    return (
        torch.matmul(a_right, a_left),
        torch.matmul(a_right, x_left) + x_right,
    )


def _pad_tree_diag(a, x):
    length = x.shape[1]
    tree_length = _next_power_of_2(length)
    if tree_length == length:
        return a, x, length
    pad_len = tree_length - length
    a_pad = torch.ones(
        a.shape[0],
        pad_len,
        a.shape[2],
        dtype=a.dtype,
        device=a.device,
    )
    x_pad = torch.zeros(
        x.shape[0],
        pad_len,
        x.shape[2],
        x.shape[3],
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((a, a_pad), dim=1), torch.cat((x, x_pad), dim=1), length


def _pad_tree_mat(a, x):
    length = x.shape[1]
    tree_length = _next_power_of_2(length)
    if tree_length == length:
        return a, x, length
    pad_len = tree_length - length
    state_dim = a.shape[-1]
    eye = torch.eye(state_dim, dtype=a.dtype, device=a.device).reshape(
        1,
        1,
        state_dim,
        state_dim,
    )
    a_pad = eye.expand(a.shape[0], pad_len, state_dim, state_dim)
    x_pad = torch.zeros(
        x.shape[0],
        pad_len,
        x.shape[2],
        x.shape[3],
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((a, a_pad), dim=1), torch.cat((x, x_pad), dim=1), length


def _pscan_blelloch_diag_flat(a, x):
    a, x, original_length = _pad_tree_diag(a, x)
    levels = [(a, x)]
    while a.shape[1] > 1:
        a_left = a[:, 0::2]
        x_left = x[:, 0::2]
        a_right = a[:, 1::2]
        x_right = x[:, 1::2]
        a, x = _combine_diag(a_left, x_left, a_right, x_right)
        levels.append((a, x))

    prefix_a = torch.ones_like(levels[-1][0])
    prefix_x = torch.zeros_like(levels[-1][1])
    for level in range(len(levels) - 2, -1, -1):
        child_a, child_x = levels[level]
        left_a = child_a[:, 0::2]
        left_x = child_x[:, 0::2]
        left_prefix_a = prefix_a
        left_prefix_x = prefix_x
        right_prefix_a, right_prefix_x = _combine_diag(
            prefix_a,
            prefix_x,
            left_a,
            left_x,
        )
        prefix_a = _interleave_tree(left_prefix_a, right_prefix_a)
        prefix_x = _interleave_tree(left_prefix_x, right_prefix_x)

    y_a, y_x = _combine_diag(prefix_a, prefix_x, levels[0][0], levels[0][1])
    return y_x[:, :original_length]


def _pscan_blelloch_mat_flat(a, x):
    a, x, original_length = _pad_tree_mat(a, x)
    levels = [(a, x)]
    while a.shape[1] > 1:
        a_left = a[:, 0::2]
        x_left = x[:, 0::2]
        a_right = a[:, 1::2]
        x_right = x[:, 1::2]
        a, x = _combine_mat(a_left, x_left, a_right, x_right)
        levels.append((a, x))

    state_dim = a.shape[-1]
    eye = torch.eye(state_dim, dtype=a.dtype, device=a.device).reshape(
        1,
        1,
        state_dim,
        state_dim,
    )
    prefix_a = eye.expand_as(levels[-1][0])
    prefix_x = torch.zeros_like(levels[-1][1])
    for level in range(len(levels) - 2, -1, -1):
        child_a, child_x = levels[level]
        left_a = child_a[:, 0::2]
        left_x = child_x[:, 0::2]
        left_prefix_a = prefix_a
        left_prefix_x = prefix_x
        right_prefix_a, right_prefix_x = _combine_mat(
            prefix_a,
            prefix_x,
            left_a,
            left_x,
        )
        prefix_a = _interleave_tree(left_prefix_a, right_prefix_a)
        prefix_x = _interleave_tree(left_prefix_x, right_prefix_x)

    y_a, y_x = _combine_mat(prefix_a, prefix_x, levels[0][0], levels[0][1])
    return y_x[:, :original_length]


def pscan_torch_tree(a, x):
    a, x, is_diag, squeeze_output = _validate_pscan_inputs(a, x)
    t_steps = x.shape[1]
    if t_steps == 1:
        return x.squeeze(-1) if squeeze_output else x
    batch, _, channels = x.shape[:3]
    batch_channels = batch * channels
    if is_diag:
        a_flat = a.permute(0, 2, 1, 3).reshape(batch_channels, t_steps, -1)
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(
            batch_channels, t_steps, -1, x.shape[-1]
        )
        y_flat = _pscan_blelloch_diag_flat(a_flat, x_flat)
        state_dim = y_flat.shape[2]
    else:
        state_dim = a.shape[-1]
        a_flat = a.permute(0, 2, 1, 3, 4).reshape(
            batch_channels, t_steps, state_dim, state_dim
        )
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(
            batch_channels, t_steps, state_dim, x.shape[-1]
        )
        y_flat = _pscan_blelloch_mat_flat(a_flat, x_flat)
    y = y_flat.reshape(batch, channels, t_steps, state_dim, x.shape[-1]).permute(
        0, 2, 1, 3, 4
    )
    return y.squeeze(-1) if squeeze_output else y


def _expand_diag_to_matrix(a_diag, state_dim):
    shape = a_diag.shape[:-1] + (state_dim, state_dim)
    a_mat = torch.zeros(shape, dtype=a_diag.dtype, device=a_diag.device)
    idx = torch.arange(state_dim, device=a_diag.device)
    a_mat[..., idx, idx] = a_diag
    return a_mat


def _triton_is_usable(a, x, is_diag):
    if triton is None or not a.is_cuda or not x.is_cuda:
        return False
    if a.dtype not in {torch.complex64, torch.complex128}:
        return False
    if x.dtype != a.dtype:
        return False
    state_dim = x.shape[-2]
    x_cols = x.shape[-1]
    if state_dim > 2 or x_cols > 2:
        return False
    if not is_diag and a.shape[-1] > 2:
        return False
    return True


def _run_mat2x2_pscan_triton(a_real, x_real, length, reverse=False):
    y_real = torch.empty_like(x_real)
    block_size = max(16, _next_power_of_2(length))
    _mat2x2_pscan_kernel[(a_real.shape[0],)](
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
        length,
        block_size,
        reverse,
    )
    return y_real


class _TritonPScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, x, is_diag):
        batch, length, channels, state_dim, x_cols = x.shape
        pad_state = state_dim == 1
        pad_cols = x_cols == 1
        x_in = x
        if pad_state:
            x_in = F.pad(x_in, (0, 0, 0, 1), mode="constant", value=0.0)
        if pad_cols:
            x_in = F.pad(x_in, (0, 1), mode="constant", value=0.0)
        a_mat = _expand_diag_to_matrix(a, state_dim) if is_diag else a
        if pad_state:
            a_mat = F.pad(a_mat, (0, 1, 0, 1), mode="constant", value=0.0)
            a_mat[..., -1, -1] = 1.0
        state_pad, cols_pad = x_in.shape[-2], x_in.shape[-1]
        a_flat = a_mat.permute(0, 2, 1, 3, 4).reshape(
            batch * channels,
            length,
            state_pad,
            state_pad,
        )
        x_flat = x_in.permute(0, 2, 1, 3, 4).reshape(
            batch * channels,
            length,
            state_pad,
            cols_pad,
        )
        a_real = torch.view_as_real(a_flat).contiguous()
        x_real = torch.view_as_real(x_flat).contiguous()
        y_real = _run_mat2x2_pscan_triton(a_real, x_real, length, reverse=False)
        y_flat = torch.view_as_complex(y_real.contiguous())
        y_saved = (
            y_flat.reshape(batch, channels, length, state_pad, cols_pad)
            .permute(
                0,
                2,
                1,
                3,
                4,
            )
            .contiguous()
        )
        y = y_saved
        if pad_state:
            y = y[..., :state_dim, :]
        if pad_cols:
            y = y[..., :x_cols]
        ctx.save_for_backward(a_mat, y_saved)
        ctx.is_diag = is_diag
        ctx.pad_state = pad_state
        ctx.pad_cols = pad_cols
        ctx.batch = batch
        ctx.length = length
        ctx.channels = channels
        ctx.state_dim = state_dim
        ctx.x_cols = x_cols
        ctx.state_pad = state_pad
        ctx.cols_pad = cols_pad
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a_mat, y_saved = ctx.saved_tensors
        batch = ctx.batch
        length = ctx.length
        channels = ctx.channels
        state_dim = ctx.state_dim
        x_cols = ctx.x_cols
        state_pad = ctx.state_pad
        cols_pad = ctx.cols_pad
        grad_in = grad_y
        if ctx.pad_state:
            grad_in = F.pad(grad_in, (0, 0, 0, 1), mode="constant", value=0.0)
        if ctx.pad_cols:
            grad_in = F.pad(grad_in, (0, 1), mode="constant", value=0.0)
        identity = (
            torch.eye(
                state_pad,
                dtype=a_mat.dtype,
                device=a_mat.device,
            )
            .reshape(1, 1, 1, state_pad, state_pad)
            .expand(
                batch,
                1,
                channels,
                state_pad,
                state_pad,
            )
        )
        a_shift = torch.cat([a_mat[:, 1:], identity], dim=1)
        a_h = a_shift.conj().transpose(-1, -2)
        a_h_flat = a_h.permute(0, 2, 1, 3, 4).reshape(
            batch * channels,
            length,
            state_pad,
            state_pad,
        )
        grad_flat = grad_in.permute(0, 2, 1, 3, 4).reshape(
            batch * channels,
            length,
            state_pad,
            cols_pad,
        )
        a_h_real = torch.view_as_real(a_h_flat.resolve_conj()).contiguous()
        grad_real = torch.view_as_real(grad_flat).contiguous()
        grad_x_real = _run_mat2x2_pscan_triton(
            a_h_real, grad_real, length, reverse=True
        )
        grad_x_flat = torch.view_as_complex(grad_x_real.contiguous())
        grad_x_full = (
            grad_x_flat.reshape(
                batch,
                channels,
                length,
                state_pad,
                cols_pad,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        grad_x = grad_x_full
        if ctx.pad_state:
            grad_x = grad_x[..., :state_dim, :]
        if ctx.pad_cols:
            grad_x = grad_x[..., :x_cols]
        y_prev = torch.cat([torch.zeros_like(y_saved[:, :1]), y_saved[:, :-1]], dim=1)
        grad_x_for_a = (
            grad_x_flat.reshape(
                batch,
                channels,
                length,
                state_pad,
                cols_pad,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        grad_a_full = torch.einsum("blcij,blckj->blcik", grad_x_for_a, y_prev.conj())
        if ctx.pad_state:
            grad_a_full = grad_a_full[..., :state_dim, :state_dim]
        grad_a = grad_a_full.diagonal(dim1=-2, dim2=-1) if ctx.is_diag else grad_a_full
        return grad_a, grad_x, None


def pscan_triton(a, x):
    a, x, is_diag, squeeze_output = _validate_pscan_inputs(a, x)
    if x.shape[1] == 1:
        return x.squeeze(-1) if squeeze_output else x
    if not _triton_is_usable(a, x, is_diag):
        raise RuntimeError("triton pscan is unavailable for this environment or shape")
    y = _TritonPScanFunction.apply(a, x, is_diag)
    return y.squeeze(-1) if squeeze_output else y


def pscan(a, x, backend="auto"):
    a_checked, x_checked, is_diag, _ = _validate_pscan_inputs(a, x)
    if backend == "torch":
        return pscan_torch_tree(a, x)
    if backend == "triton":
        return pscan_triton(a, x)
    if backend != "auto":
        raise ValueError(f"unsupported_pscan_backend={backend}")
    if x_checked.shape[1] > 1 and _triton_is_usable(a_checked, x_checked, is_diag):
        try:
            return pscan_triton(a, x)
        except Exception:
            return pscan_torch_tree(a, x)
    return pscan_torch_tree(a, x)
