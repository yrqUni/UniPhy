import torch
import torch.nn.functional as F

import triton
import triton.language as tl

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 16}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
]


@triton.jit
def _scalar_combine(ar, ai, xr, xi, br, bi, yr, yi):
    cr = br * ar - bi * ai
    ci = br * ai + bi * ar
    zr = br * xr - bi * xi + yr
    zi = br * xi + bi * xr + yi
    return cr, ci, zr, zi


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["L"])
@triton.jit
def _scalar_pscan_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_a_batch, stride_a_time,
    stride_x_batch, stride_x_time,
    L, BLOCK_SIZE: tl.constexpr, REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_a_batch
    X_base = X_ptr + pid * stride_x_batch
    Y_base = Y_ptr + pid * stride_x_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    t = (L - 1 - offs) if REVERSE else offs

    ar = tl.load(A_base + t * stride_a_time + 0, mask=mask, other=1.0)
    ai = tl.load(A_base + t * stride_a_time + 1, mask=mask, other=0.0)
    xr = tl.load(X_base + t * stride_x_time + 0, mask=mask, other=0.0)
    xi = tl.load(X_base + t * stride_x_time + 1, mask=mask, other=0.0)

    _, _, yr, yi = tl.associative_scan(
        (ar, ai, xr, xi), axis=0, combine_fn=_scalar_combine,
    )

    tl.store(Y_base + t * stride_x_time + 0, yr, mask=mask)
    tl.store(Y_base + t * stride_x_time + 1, yi, mask=mask)


_MAT_CHUNK_SIZE = 8


@triton.jit
def _mat2x2_prod_combine(
    a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
    b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i,
):
    """Combine two 2x2 complex matrices: result = B @ A."""
    c00r = b00r * a00r - b00i * a00i + b01r * a10r - b01i * a10i
    c00i = b00r * a00i + b00i * a00r + b01r * a10i + b01i * a10r
    c01r = b00r * a01r - b00i * a01i + b01r * a11r - b01i * a11i
    c01i = b00r * a01i + b00i * a01r + b01r * a11i + b01i * a11r
    c10r = b10r * a00r - b10i * a00i + b11r * a10r - b11i * a10i
    c10i = b10r * a00i + b10i * a00r + b11r * a10i + b11i * a10r
    c11r = b10r * a01r - b10i * a01i + b11r * a11r - b11i * a11i
    c11i = b10r * a01i + b10i * a01r + b11r * a11i + b11i * a11r
    return c00r, c00i, c01r, c01i, c10r, c10i, c11r, c11i


@triton.jit
def _mat2x2_apply(
    a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
    y00r, y00i, y01r, y01i, y10r, y10i, y11r, y11i,
):
    """Compute A @ Y."""
    z00r = a00r * y00r - a00i * y00i + a01r * y10r - a01i * y10i
    z00i = a00r * y00i + a00i * y00r + a01r * y10i + a01i * y10r
    z01r = a00r * y01r - a00i * y01i + a01r * y11r - a01i * y11i
    z01i = a00r * y01i + a00i * y01r + a01r * y11i + a01i * y11r
    z10r = a10r * y00r - a10i * y00i + a11r * y10r - a11i * y10i
    z10i = a10r * y00i + a10i * y00r + a11r * y10i + a11i * y10r
    z11r = a10r * y01r - a10i * y01i + a11r * y11r - a11i * y11i
    z11i = a10r * y01i + a10i * y01r + a11r * y11i + a11i * y11r
    return z00r, z00i, z01r, z01i, z10r, z10i, z11r, z11i


@triton.jit
def _affine_combine(
    # left element: (A_l, y_l)  — A: 8 floats, y: 4 floats (2x1 col vec)
    a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
    y0r, y0i, y1r, y1i,
    # right element: (A_r, y_r)
    b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i,
    z0r, z0i, z1r, z1i,
):
    """
    Combine two affine maps (A_l, y_l) and (A_r, y_r).
    Result = (A_r @ A_l, A_r @ y_l + y_r).
    This is the associative operator for the scan Y[t] = A[t]@Y[t-1] + X[t].
    """
    # c = A_r @ A_l
    c00r, c00i, c01r, c01i, c10r, c10i, c11r, c11i = _mat2x2_prod_combine(
        a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
        b00r, b00i, b01r, b01i, b10r, b10i, b11r, b11i,
    )
    # w = A_r @ y_l + y_r
    w0r = b00r * y0r - b00i * y0i + b01r * y1r - b01i * y1i + z0r
    w0i = b00r * y0i + b00i * y0r + b01r * y1i + b01i * y1r + z0i
    w1r = b10r * y0r - b10i * y0i + b11r * y1r - b11i * y1i + z1r
    w1i = b10r * y0i + b10i * y0r + b11r * y1i + b11i * y1r + z1i
    return c00r, c00i, c01r, c01i, c10r, c10i, c11r, c11i, w0r, w0i, w1r, w1i


@triton.jit
def _mat2x2_chunk_pass1_kernel(
    A_ptr, X_ptr, Y_ptr, State_ptr,
    stride_a_batch, stride_a_time, stride_a_d1, stride_a_d2,
    stride_x_batch, stride_x_time, stride_x_d1, stride_x_d2,
    stride_s_batch, stride_s_chunk,
    L, CHUNK: tl.constexpr, REVERSE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
):
    """
    Pass 1: sequential local scan within each chunk, starting from Y=0.

    Writes:
      - Y_local[t] for all valid t (local scan result, prefix state = identity/0)
      - State[chunk] = (A_prod, Y_end) for pass 2
        A_prod = product of A values in chunk (8 floats)
        Y_end  = Y_local at chunk end (4 floats)
    State layout: [a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
                   y0r,  y0i,  y1r,  y1i]  -> 12 floats per chunk

    grid = (BC * N_CHUNKS,)
    """
    pid = tl.program_id(axis=0)
    batch_id = pid // N_CHUNKS
    chunk_id = pid % N_CHUNKS

    A_base = A_ptr + batch_id * stride_a_batch
    X_base = X_ptr + batch_id * stride_x_batch
    Y_base = Y_ptr + batch_id * stride_x_batch
    S_base = State_ptr + batch_id * stride_s_batch

    chunk_start = chunk_id * CHUNK if not REVERSE else (N_CHUNKS - 1 - chunk_id) * CHUNK

    # Running state: A_acc (2x2 identity), y_acc (2x1 zero)
    acc_a00r = 1.0; acc_a00i = 0.0
    acc_a01r = 0.0; acc_a01i = 0.0
    acc_a10r = 0.0; acc_a10i = 0.0
    acc_a11r = 1.0; acc_a11i = 0.0
    acc_y0r = 0.0; acc_y0i = 0.0
    acc_y1r = 0.0; acc_y1i = 0.0

    for step in tl.static_range(CHUNK):
        t = chunk_start + step if not REVERSE else chunk_start + (CHUNK - 1 - step)
        valid = t < L

        ta = t * stride_a_time
        a00r = tl.load(A_base + ta + 0, mask=valid, other=1.0)
        a00i = tl.load(A_base + ta + 1, mask=valid, other=0.0)
        a01r = tl.load(A_base + ta + stride_a_d2 + 0, mask=valid, other=0.0)
        a01i = tl.load(A_base + ta + stride_a_d2 + 1, mask=valid, other=0.0)
        a10r = tl.load(A_base + ta + stride_a_d1 + 0, mask=valid, other=0.0)
        a10i = tl.load(A_base + ta + stride_a_d1 + 1, mask=valid, other=0.0)
        a11r = tl.load(A_base + ta + stride_a_d1 + stride_a_d2 + 0, mask=valid, other=1.0)
        a11i = tl.load(A_base + ta + stride_a_d1 + stride_a_d2 + 1, mask=valid, other=0.0)

        tx = t * stride_x_time
        x0r = tl.load(X_base + tx + 0, mask=valid, other=0.0)
        x0i = tl.load(X_base + tx + 1, mask=valid, other=0.0)
        x1r = tl.load(X_base + tx + stride_x_d1 + 0, mask=valid, other=0.0)
        x1i = tl.load(X_base + tx + stride_x_d1 + 1, mask=valid, other=0.0)

        # new_acc = affine_combine(acc, (A[t], X[t]))
        # = (A[t] @ A_acc, A[t] @ y_acc + X[t])
        new_a00r, new_a00i, new_a01r, new_a01i, new_a10r, new_a10i, new_a11r, new_a11i, \
        new_y0r, new_y0i, new_y1r, new_y1i = _affine_combine(
            acc_a00r, acc_a00i, acc_a01r, acc_a01i,
            acc_a10r, acc_a10i, acc_a11r, acc_a11i,
            acc_y0r, acc_y0i, acc_y1r, acc_y1i,
            a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i,
            x0r, x0i, x1r, x1i,
        )

        if valid:
            tl.store(Y_base + tx + 0, new_y0r)
            tl.store(Y_base + tx + 1, new_y0i)
            tl.store(Y_base + tx + stride_x_d1 + 0, new_y1r)
            tl.store(Y_base + tx + stride_x_d1 + 1, new_y1i)

        acc_a00r = new_a00r if valid else acc_a00r
        acc_a00i = new_a00i if valid else acc_a00i
        acc_a01r = new_a01r if valid else acc_a01r
        acc_a01i = new_a01i if valid else acc_a01i
        acc_a10r = new_a10r if valid else acc_a10r
        acc_a10i = new_a10i if valid else acc_a10i
        acc_a11r = new_a11r if valid else acc_a11r
        acc_a11i = new_a11i if valid else acc_a11i
        acc_y0r = new_y0r if valid else acc_y0r
        acc_y0i = new_y0i if valid else acc_y0i
        acc_y1r = new_y1r if valid else acc_y1r
        acc_y1i = new_y1i if valid else acc_y1i

    # Write chunk-end state to State buffer
    s = chunk_id * stride_s_chunk
    tl.store(S_base + s + 0,  acc_a00r)
    tl.store(S_base + s + 1,  acc_a00i)
    tl.store(S_base + s + 2,  acc_a01r)
    tl.store(S_base + s + 3,  acc_a01i)
    tl.store(S_base + s + 4,  acc_a10r)
    tl.store(S_base + s + 5,  acc_a10i)
    tl.store(S_base + s + 6,  acc_a11r)
    tl.store(S_base + s + 7,  acc_a11i)
    tl.store(S_base + s + 8,  acc_y0r)
    tl.store(S_base + s + 9,  acc_y0i)
    tl.store(S_base + s + 10, acc_y1r)
    tl.store(S_base + s + 11, acc_y1i)


@triton.jit
def _mat2x2_chunk_pass2_kernel(
    State_ptr,
    stride_s_batch,
    stride_s_chunk,
    N_CHUNKS: tl.constexpr,
    N_CHUNKS_POW2: tl.constexpr,
):
    """
    Pass 2: prefix scan over State[chunk] using _affine_combine.

    After this pass, State[k] contains the prefix affine map that maps
    chunk k's local Y_local values to the global Y values:
        global_Y[t in chunk k] = A_prefix[k-1] @ Y_local[t] + Y_carry[k-1]
    where (A_prefix[k-1], Y_carry[k-1]) = State[k-1] after this scan.

    State layout per chunk: 12 floats (A: 8, Y: 4).
    grid = (BC,)
    """
    pid = tl.program_id(axis=0)
    S_base = State_ptr + pid * stride_s_batch

    offs = tl.arange(0, N_CHUNKS_POW2)
    mask = offs < N_CHUNKS
    s = offs * stride_s_chunk

    a00r = tl.load(S_base + s + 0,  mask=mask, other=1.0)
    a00i = tl.load(S_base + s + 1,  mask=mask, other=0.0)
    a01r = tl.load(S_base + s + 2,  mask=mask, other=0.0)
    a01i = tl.load(S_base + s + 3,  mask=mask, other=0.0)
    a10r = tl.load(S_base + s + 4,  mask=mask, other=0.0)
    a10i = tl.load(S_base + s + 5,  mask=mask, other=0.0)
    a11r = tl.load(S_base + s + 6,  mask=mask, other=1.0)
    a11i = tl.load(S_base + s + 7,  mask=mask, other=0.0)
    y0r  = tl.load(S_base + s + 8,  mask=mask, other=0.0)
    y0i  = tl.load(S_base + s + 9,  mask=mask, other=0.0)
    y1r  = tl.load(S_base + s + 10, mask=mask, other=0.0)
    y1i  = tl.load(S_base + s + 11, mask=mask, other=0.0)

    (
        pa00r, pa00i, pa01r, pa01i, pa10r, pa10i, pa11r, pa11i,
        py0r, py0i, py1r, py1i,
    ) = tl.associative_scan(
        (a00r, a00i, a01r, a01i, a10r, a10i, a11r, a11i, y0r, y0i, y1r, y1i),
        axis=0,
        combine_fn=_affine_combine,
    )

    tl.store(S_base + s + 0,  pa00r, mask=mask)
    tl.store(S_base + s + 1,  pa00i, mask=mask)
    tl.store(S_base + s + 2,  pa01r, mask=mask)
    tl.store(S_base + s + 3,  pa01i, mask=mask)
    tl.store(S_base + s + 4,  pa10r, mask=mask)
    tl.store(S_base + s + 5,  pa10i, mask=mask)
    tl.store(S_base + s + 6,  pa11r, mask=mask)
    tl.store(S_base + s + 7,  pa11i, mask=mask)
    tl.store(S_base + s + 8,  py0r,  mask=mask)
    tl.store(S_base + s + 9,  py0i,  mask=mask)
    tl.store(S_base + s + 10, py1r,  mask=mask)
    tl.store(S_base + s + 11, py1i,  mask=mask)


@triton.jit
def _mat2x2_chunk_pass3_kernel(
    Y_ptr, State_ptr,
    stride_x_batch, stride_x_time, stride_x_d1,
    stride_s_batch, stride_s_chunk,
    L, CHUNK: tl.constexpr, REVERSE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
):
    """
    Pass 3: apply prefix state from chunk k-1 to every element in chunk k.

    For chunk k (k > 0):
        Y[t] = A_prefix[k-1] @ Y_local[t] + Y_carry[k-1]
    where (A_prefix[k-1], Y_carry[k-1]) = State[k-1] after pass 2.

    grid = (BC * N_CHUNKS,)
    """
    pid = tl.program_id(axis=0)
    batch_id = pid // N_CHUNKS
    chunk_id = pid % N_CHUNKS

    if chunk_id == 0:
        return

    Y_base = Y_ptr + batch_id * stride_x_batch
    S_base = State_ptr + batch_id * stride_s_batch

    # Load prefix state from chunk k-1
    s = (chunk_id - 1) * stride_s_chunk
    p00r = tl.load(S_base + s + 0)
    p00i = tl.load(S_base + s + 1)
    p01r = tl.load(S_base + s + 2)
    p01i = tl.load(S_base + s + 3)
    p10r = tl.load(S_base + s + 4)
    p10i = tl.load(S_base + s + 5)
    p11r = tl.load(S_base + s + 6)
    p11i = tl.load(S_base + s + 7)
    c0r  = tl.load(S_base + s + 8)
    c0i  = tl.load(S_base + s + 9)
    c1r  = tl.load(S_base + s + 10)
    c1i  = tl.load(S_base + s + 11)

    chunk_start = chunk_id * CHUNK if not REVERSE else (N_CHUNKS - 1 - chunk_id) * CHUNK

    for step in tl.static_range(CHUNK):
        t = chunk_start + step if not REVERSE else chunk_start + (CHUNK - 1 - step)
        valid = t < L

        if valid:
            tx = t * stride_x_time
            # Load Y_local[t]
            yl0r = tl.load(Y_base + tx + 0)
            yl0i = tl.load(Y_base + tx + 1)
            yl1r = tl.load(Y_base + tx + stride_x_d1 + 0)
            yl1i = tl.load(Y_base + tx + stride_x_d1 + 1)

            # Y[t] = A_prefix @ Y_local[t] + Y_carry
            z0r = p00r * yl0r - p00i * yl0i + p01r * yl1r - p01i * yl1i + c0r
            z0i = p00r * yl0i + p00i * yl0r + p01r * yl1i + p01i * yl1r + c0i
            z1r = p10r * yl0r - p10i * yl0i + p11r * yl1r - p11i * yl1i + c1r
            z1i = p10r * yl0i + p10i * yl0r + p11r * yl1i + p11i * yl1r + c1i

            tl.store(Y_base + tx + 0, z0r)
            tl.store(Y_base + tx + 1, z0i)
            tl.store(Y_base + tx + stride_x_d1 + 0, z1r)
            tl.store(Y_base + tx + stride_x_d1 + 1, z1i)


def _run_scalar_pscan(A_real, X_real, L, reverse=False):
    Y_real = torch.empty_like(X_real)
    _scalar_pscan_kernel[(A_real.shape[0],)](
        A_real, X_real, Y_real,
        A_real.stride(0), A_real.stride(1),
        X_real.stride(0), X_real.stride(1),
        L, REVERSE=reverse,
    )
    return Y_real


def _run_mat2x2_pscan(A_real, X_real, L, reverse=False):
    """
    Three-pass chunk scan for 2x2 complex matrix SSM.

    Pass 1: local sequential scan per chunk starting from (I, 0).
            Stores Y_local and chunk-end state (A_prod, Y_end).
    Pass 2: prefix scan over chunk states using _affine_combine.
            After pass 2, State[k] = (A_prefix_k, Y_carry_k).
    Pass 3: correct each non-first chunk:
            Y[t] = A_prefix[k-1] @ Y_local[t] + Y_carry[k-1].

    State tensor: [BC, n_chunks, 12] float32
    """
    BC = A_real.shape[0]
    n_chunks = (L + _MAT_CHUNK_SIZE - 1) // _MAT_CHUNK_SIZE
    n_chunks_pow2 = 1
    while n_chunks_pow2 < n_chunks:
        n_chunks_pow2 *= 2

    Y_real = torch.empty_like(X_real)
    # 12 floats per chunk: 8 for A, 4 for Y (2x1 vector)
    State = torch.empty(BC, n_chunks, 12, dtype=torch.float32, device=A_real.device)

    _mat2x2_chunk_pass1_kernel[(BC * n_chunks,)](
        A_real, X_real, Y_real, State,
        A_real.stride(0), A_real.stride(1),
        A_real.stride(2), A_real.stride(3),
        X_real.stride(0), X_real.stride(1),
        X_real.stride(2), X_real.stride(3),
        State.stride(0), State.stride(1),
        L, CHUNK=_MAT_CHUNK_SIZE, REVERSE=reverse, N_CHUNKS=n_chunks,
    )

    if n_chunks > 1:
        _mat2x2_chunk_pass2_kernel[(BC,)](
            State,
            State.stride(0), State.stride(1),
            N_CHUNKS=n_chunks, N_CHUNKS_POW2=n_chunks_pow2,
        )

        _mat2x2_chunk_pass3_kernel[(BC * n_chunks,)](
            Y_real, State,
            X_real.stride(0), X_real.stride(1),
            X_real.stride(2),
            State.stride(0), State.stride(1),
            L, CHUNK=_MAT_CHUNK_SIZE, REVERSE=reverse, N_CHUNKS=n_chunks,
        )

    return Y_real


def _run_mat2x2_pscan_torch(A_complex, X_complex, reverse=False):
    B, L, D1, D2 = X_complex.shape
    Y = torch.zeros_like(X_complex)
    acc = torch.zeros(B, D1, D2, dtype=X_complex.dtype, device=X_complex.device)
    if reverse:
        for i in range(L - 1, -1, -1):
            acc = torch.bmm(A_complex[:, i].conj().transpose(-1, -2), acc)
            acc = acc + X_complex[:, i]
            Y[:, i] = acc
    else:
        for i in range(L):
            acc = torch.bmm(A_complex[:, i], acc) + X_complex[:, i]
            Y[:, i] = acc
    return Y


def _expand_diag(A_diag, D):
    shape = A_diag.shape[:-1] + (D, D)
    A_mat = torch.zeros(shape, dtype=A_diag.dtype, device=A_diag.device)
    idx = torch.arange(D, device=A_diag.device)
    A_mat[..., idx, idx] = A_diag
    return A_mat


class _ScalarPScanFunction(torch.autograd.Function):
    """
    Parallel prefix scan for scalar complex SSM: Y[t] = A[t]*Y[t-1] + X[t].

    Shapes:
        A : [B, L, C]  complex64  — per-step decay scalars
        X : [B, L, C]  complex64  — per-step inputs
        Y : [B, L, C]  complex64  — prefix-scan outputs

    The combine operator is (c, z) = (b*a, b*x + y), which corresponds to
    composing two affine maps y -> a*y + x and y -> b*y + z.

    Backward pass uses a reverse scan with the conjugate-shifted decay:
        grad_X[t] = sum_{s>=t}  (prod_{k=t+1}^{s} A[k])  *  grad_Y[s]
    which is itself a reverse scan with A_shift[t] = conj(A[t+1]).

    Wirtinger gradient for A:
        grad_A[t] = grad_X[t] * conj(Y[t-1])
    """

    @staticmethod
    def forward(ctx, A, X):
        B, L, C = X.shape
        A_flat = A.reshape(B * C, L)
        X_flat = X.reshape(B * C, L)
        A_real = torch.view_as_real(A_flat.contiguous()).contiguous()
        X_real = torch.view_as_real(X_flat.contiguous()).contiguous()
        Y_real = _run_scalar_pscan(A_real, X_real, L)
        Y = torch.view_as_complex(Y_real.contiguous()).reshape(B, L, C)
        ctx.save_for_backward(A, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        A, Y = ctx.saved_tensors
        B, L, C = Y.shape

        Y_prev = torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
        ones = torch.ones(B, 1, C, dtype=A.dtype, device=A.device)
        A_shift = torch.cat([A[:, 1:].conj(), ones], dim=1)

        A_flat = A_shift.reshape(B * C, L)
        G_flat = grad_Y.reshape(B * C, L)
        A_real = torch.view_as_real(A_flat.contiguous()).contiguous()
        G_real = torch.view_as_real(G_flat.contiguous()).contiguous()
        grad_X_real = _run_scalar_pscan(A_real, G_real, L, reverse=True)
        grad_X = torch.view_as_complex(
            grad_X_real.contiguous()
        ).reshape(B, L, C)

        grad_A = grad_X * Y_prev.conj()
        return grad_A, grad_X


class _MatPScanFunction(torch.autograd.Function):
    """
    Parallel prefix scan for 2x2 complex matrix SSM: Y[t] = A[t]@Y[t-1] + X[t].

    Shapes (before internal padding):
        A : [B, L, C, D1, D1]  complex64  — transition matrices (or diag vector)
        X : [B, L, C, D1, D2]  complex64  — inputs
        Y : [B, L, C, D1, D2]  complex64  — outputs

    When is_diag=True, A has shape [B, L, C, D1] and is expanded to diagonal
    matrices before the scan.  D1 must equal 1 or 2; D2 must equal 1 or 2.
    Inputs are zero-padded to D1=D2=2 if needed and trimmed on output.

    The forward scan uses the three-pass chunk algorithm implemented in
    _run_mat2x2_pscan.  Backward uses the same kernel with REVERSE=True and
    the conjugate-transposed shifted A matrix.
    """

    @staticmethod
    def forward(ctx, A, X, is_diag):
        B, L, C, D1, D2_x = X.shape
        pad_d2 = D2_x == 1
        pad_d1 = D1 == 1

        ctx.is_diag = is_diag
        ctx.pad_d2 = pad_d2
        ctx.pad_d1 = pad_d1
        ctx.D1_orig = D1
        ctx.D2_orig = D2_x

        X_in = X
        if pad_d1:
            X_in = F.pad(X_in, (0, 0, 0, 1), value=0.0)
        if pad_d2:
            X_in = F.pad(X_in, (0, 1), value=0.0)

        A_mat = _expand_diag(A, D1) if is_diag else A
        if pad_d1:
            A_mat = F.pad(A_mat, (0, 1, 0, 1), value=0.0)
            A_mat[..., -1, -1] = 1.0

        D1_pad, D2_pad = X_in.shape[-2], X_in.shape[-1]

        A_flat = A_mat.permute(0, 2, 1, 3, 4).reshape(
            B * C, L, A_mat.shape[-2], A_mat.shape[-1],
        )
        X_flat = X_in.permute(0, 2, 1, 3, 4).reshape(B * C, L, D1_pad, D2_pad)
        A_real = torch.view_as_real(A_flat.contiguous()).contiguous()
        X_real = torch.view_as_real(X_flat.contiguous()).contiguous()

        Y_real = _run_mat2x2_pscan(A_real, X_real, L)

        Y_flat = torch.view_as_complex(Y_real.contiguous())
        Y_full = (
            Y_flat.reshape(B, C, L, D1_pad, D2_pad)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        if pad_d1:
            Y_full = Y_full[..., :D1, :]
        if pad_d2:
            Y_full = Y_full[..., :D2_x]

        Y_saved = (
            Y_flat.reshape(B, C, L, D1_pad, D2_pad)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        ctx.save_for_backward(A_mat, Y_saved)
        ctx.B, ctx.L, ctx.C = B, L, C
        ctx.D1_pad, ctx.D2_pad = D1_pad, D2_pad
        return Y_full

    @staticmethod
    def backward(ctx, grad_Y):
        A_mat, Y_saved = ctx.saved_tensors
        B, L, C = ctx.B, ctx.L, ctx.C
        D1_pad, D2_pad = ctx.D1_pad, ctx.D2_pad
        D1_orig, D2_orig = ctx.D1_orig, ctx.D2_orig

        grad_Y_in = grad_Y
        if ctx.pad_d1:
            grad_Y_in = F.pad(grad_Y_in, (0, 0, 0, 1), value=0.0)
        if ctx.pad_d2:
            grad_Y_in = F.pad(grad_Y_in, (0, 1), value=0.0)

        D1 = A_mat.shape[-2]
        I = (
            torch.eye(D1, dtype=A_mat.dtype, device=A_mat.device)
            .reshape(1, 1, 1, D1, D1)
            .expand(B, 1, C, D1, D1)
        )
        A_shift = torch.cat([A_mat[:, 1:], I], dim=1)
        A_H = A_shift.conj().transpose(-1, -2)

        A_H_flat = A_H.permute(0, 2, 1, 3, 4).reshape(B * C, L, D1, D1)
        gY_flat = grad_Y_in.permute(0, 2, 1, 3, 4).reshape(
            B * C, L, D1_pad, D2_pad,
        )
        A_H_real = torch.view_as_real(A_H_flat.resolve_conj().contiguous()).contiguous()
        gY_real = torch.view_as_real(gY_flat.contiguous()).contiguous()

        gX_real = _run_mat2x2_pscan(A_H_real, gY_real, L, reverse=True)

        gX_flat = torch.view_as_complex(gX_real.contiguous())
        gX_full = (
            gX_flat.reshape(B, C, L, D1_pad, D2_pad)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        if ctx.pad_d1:
            gX_full = gX_full[..., :D1_orig, :]
        grad_X = gX_full[..., :D2_orig] if ctx.pad_d2 else gX_full

        Y_prev = torch.cat(
            [torch.zeros_like(Y_saved[:, :1]), Y_saved[:, :-1]], dim=1,
        )
        gX_for_A = (
            gX_flat.reshape(B, C, L, D1_pad, D2_pad)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        grad_A_full = torch.einsum(
            "blcij,blckj->blcik", gX_for_A, Y_prev.conj(),
        )
        if ctx.pad_d1:
            grad_A_full = grad_A_full[..., :D1_orig, :D1_orig]

        grad_A = (
            grad_A_full.diagonal(dim1=-2, dim2=-1)
            if ctx.is_diag
            else grad_A_full
        )
        return grad_A, grad_X, None


def pscan(A, X):
    """
    Complex parallel prefix scan: Y[t] = A[t] @ Y[t-1] + X[t], Y[-1] = 0.

    Dispatch rules (determined by A.ndim):
        A.ndim == 4  ->  diagonal mode  [B, L, C, D1]
        A.ndim == 5  ->  matrix mode    [B, L, C, D1, D1]

    When X.ndim == 4 the trailing singleton dimension is added automatically
    and squeezed from the output.

    Scalar path (diagonal mode, D1 == 1): single-kernel associative scan with
    8 float32 values in the combine function.

    Matrix path (diagonal mode D1 == 2, or full matrix mode): three-pass chunk
    scan.  Pass 1 runs a local sequential scan per chunk from (I, 0).  Pass 2
    does a prefix scan over (A_prod, Y_end) states using the affine combine
    operator (B,y_B)∘(A,y_A)=(B@A, B@y_A+y_B).  Pass 3 corrects each
    non-first chunk: Y[t] = A_prefix[k-1]@Y_local[t] + Y_carry[k-1].
    """
    squeeze_output = X.ndim == 4
    if squeeze_output:
        X = X.unsqueeze(-1)

    is_diag = A.ndim == 4

    if is_diag and X.shape[-2] == 1 and X.shape[-1] == 1:
        B, L, C, _, _ = X.shape
        Y_sc = _ScalarPScanFunction.apply(A[..., 0], X[..., 0, 0])
        Y = Y_sc.unsqueeze(-1).unsqueeze(-1)
        return Y.squeeze(-1) if squeeze_output else Y

    Y = _MatPScanFunction.apply(A, X, is_diag)
    return Y.squeeze(-1) if squeeze_output else Y
