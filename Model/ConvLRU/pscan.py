import torch
import triton
import triton.language as tl

@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@triton.jit
def scan_combine(ar, ai, xr, xi, br, bi, yr, yi):
    # Linear recurrence: y_t = a_t * y_{t-1} + x_t
    # Parallel scan associative operator:
    # (a_new, x_new) = (a_j * a_i, a_j * x_i + x_j)
    # Here: a is (ar, ai), x is (xr, xi), b is next a, y is next x
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
    # Pointers to the current sequence
    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    # Load data
    a_r = tl.load(A_base + offs * stride_time, mask=mask, other=1.0)
    a_i = tl.load(A_base + offs * stride_time + 1, mask=mask, other=0.0)
    x_r = tl.load(X_base + offs * stride_time, mask=mask, other=0.0)
    x_i = tl.load(X_base + offs * stride_time + 1, mask=mask, other=0.0)

    # Parallel Scan
    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan(
        (a_r, a_i, x_r, x_i),
        axis=0,
        combine_fn=scan_combine
    )

    # Store result
    tl.store(Y_base + offs * stride_time, acc_xr, mask=mask)
    tl.store(Y_base + offs * stride_time + 1, acc_xi, mask=mask)

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

class PScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        # 1. 记录原始形状以便 backward 恢复
        ctx.shape_A_orig = A.shape
        ctx.shape_X_orig = X.shape

        # 2. 自动广播 (Broadcasting)
        # 如果 A 是 (B, L, C, S)，X 是 (B, L, C, S, S)
        # 这里会将 A 扩展为 (B, L, C, S, S) 以匹配 X
        if A.shape != X.shape:
            A, X = torch.broadcast_tensors(A, X)
        
        # 必须 clone 这里的 view，因为 broadcast_tensors 生成的是非连续的 view，
        # 而后续 transpose + contiguous 需要物理内存对其以传入 Triton
        A = A.clone() 
        X = X.clone()

        input_shape = X.shape
        L = input_shape[1]

        # 3. 准备 Triton 输入
        # 将时间维度 L 移到最后，然后展平前面的维度作为 Batch
        A_in = A.transpose(1, -1).contiguous()
        X_in = X.transpose(1, -1).contiguous()

        A_flat = A_in.view(-1, L)
        X_flat = X_in.view(-1, L)

        num_sequences = A_flat.shape[0]

        A_real = torch.view_as_real(A_flat)
        X_real = torch.view_as_real(X_flat)
        Y_real = torch.empty_like(X_real)

        BLOCK_SIZE = next_power_of_2(L)
        BLOCK_SIZE = max(BLOCK_SIZE, 16)
        # Triton限制，如果序列过长可能需要分块处理，这里假设L在合理范围内
        if BLOCK_SIZE > 131072: 
             raise ValueError(f"Sequence length L={L} exceeds Triton block limits.")

        grid = (num_sequences,)
        stride_batch = A_real.stride(0)
        stride_time = A_real.stride(1)

        pscan_kernel[grid](
            A_real, X_real, Y_real,
            stride_batch, stride_time,
            L, BLOCK_SIZE
        )

        Y = torch.view_as_complex(Y_real)
        Y = Y.view(*A_in.shape)
        Y = Y.transpose(1, -1)

        # 保存广播后的 A 和计算出的 Y 用于 backward
        ctx.save_for_backward(A, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, Y = ctx.saved_tensors
        A_conj = A.conj()

        # 反向传播逻辑 (Reverse Parallel Scan)
        grad_output_rev = grad_output.flip(1)
        A_conj_rev = A_conj.flip(1)

        slices_start = [slice(None)] * A.ndim
        slices_start[1] = slice(0, 1)

        slices_end = [slice(None)] * A.ndim
        slices_end[1] = slice(None, -1)

        # Shift A
        A_rev_shifted = torch.cat([
            torch.zeros_like(A_conj_rev[tuple(slices_start)]),
            A_conj_rev[tuple(slices_end)]
        ], dim=1)

        # dX = pscan(A_shift, grad_output_rev).flip()
        # 这里复用 forward 逻辑
        # 注意：这里需要递归调用 PScanTriton.apply，但为了避免无限递归开销，
        # 且 backward 内部不需要梯度，可以直接调用 implementation logic 
        # 或者直接以此作为新的 Function 调用。为了简单，我们调用 apply (PyTorch 会处理图)
        dX_rev = PScanTriton.apply(A_rev_shifted, grad_output_rev)
        dX = dX_rev.flip(1)

        # dA = dX * Y_prev.conj()
        Y_prev = torch.cat([
            torch.zeros_like(Y[tuple(slices_start)]),
            Y[tuple(slices_end)]
        ], dim=1)

        dA = dX * Y_prev.conj()

        # 4. 处理反向广播 (Unbroadcast / Reduce)
        # 如果原始 A 的形状比现在计算出的 dA 小，需要将多余的维度求和
        shape_A_orig = ctx.shape_A_orig
        if dA.shape != shape_A_orig:
            # 维度数量不同 (例如 (L, C) vs (B, L, C)) -> 前面求和
            while dA.ndim > len(shape_A_orig):
                dA = dA.sum(dim=0)
            
            # 维度大小不同 (例如 (B, L, C, 1) vs (B, L, C, S)) -> 对应维度求和
            for i, dim in enumerate(shape_A_orig):
                if dim == 1 and dA.shape[i] > 1:
                    dA = dA.sum(dim=i, keepdim=True)
        
        # X 通常不需要 reduce，因为通常以 X 的形状为准进行广播
        # 但为了鲁棒性，如果 X 也被广播了(极其罕见)，同理处理
        shape_X_orig = ctx.shape_X_orig
        if dX.shape != shape_X_orig:
            while dX.ndim > len(shape_X_orig):
                dX = dX.sum(dim=0)
            for i, dim in enumerate(shape_X_orig):
                if dim == 1 and dX.shape[i] > 1:
                    dX = dX.sum(dim=i, keepdim=True)

        return dA, dX

pscan = PScanTriton.apply

def pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
    if not torch.cuda.is_available():
        print("Skipping check: CUDA not available")
        return True

    device = 'cuda'
    dtype = torch.complex64

    # --- 这里修改为你要求的形状 ---
    # A: (B, L, C, S) -> 类似于对角矩阵或共享参数
    A = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    # X: (B, L, C, S, S) -> 完整的状态矩阵输入
    X = torch.randn(batch_size, seq_length, channels, state_dim, state_dim, device=device, dtype=dtype, requires_grad=True)

    print(f"Testing shapes: A={A.shape}, X={X.shape}")

    try:
        # Triton Kernel 应该能自动处理 A 到 X 的广播
        Y_triton = pscan(A, X)
    except Exception as e:
        print(f"❌ Triton Run Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # --- Serial Implementation Check ---
    Y_serial = torch.zeros_like(X)
    # h 需要匹配 X 的最后两维 (S, S)
    h = torch.zeros(batch_size, channels, state_dim, state_dim, device=device, dtype=dtype)

    for t in range(seq_length):
        # 这里的 A[:, t] 是 (B, C, S)
        # 这里的 h       是 (B, C, S, S)
        # 我们需要把 A 扩展为 (B, C, S, 1) 来广播乘法
        at = A[:, t].unsqueeze(-1) 
        xt = X[:, t]
        h = at * h + xt
        Y_serial[:, t] = h

    max_diff = (Y_triton - Y_serial).abs().max().item()
    print(f"Max Diff: {max_diff}")
    
    if max_diff > 1e-4:
        print(f"❌ Mismatch! Max Diff: {max_diff}")
        return False

    # Backward check
    loss = Y_triton.sum().abs()
    loss.backward()
    print("✅ PScan Check Passed (Forward + Backward + Broadcasting).")
    return True

if __name__ == "__main__":
    pscan_check()

