import torch
import triton
import triton.language as tl

def get_autotune_configs():
    return [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

@triton.jit
def matmul_r(a, b, R: tl.constexpr):
    acc = tl.zeros((R, R), dtype=tl.float32)
    for k in range(R):
        acc += a[:, k, None] * b[None, k, :]
    return acc

@triton.jit
def matmul_rx(a, x, R: tl.constexpr):
    acc = tl.zeros((R, R), dtype=tl.float32)
    for k in range(R):
        acc += a[:, k, None] * x[None, k, :]
    return acc

@triton.jit
def diag_mul_rx(a, x, R: tl.constexpr):
    return a[:, None] * x

@triton.jit
def combine_matrix(a_curr, x_curr, a_prev, x_prev, R: tl.constexpr):
    a_new = matmul_r(a_curr, a_prev, R)
    x_tmp = matmul_rx(a_curr, x_prev, R)
    x_new = x_tmp + x_curr
    return a_new, x_new

@triton.jit
def combine_diag(a_curr, x_curr, a_prev, x_prev, R: tl.constexpr):
    a_new = a_curr * a_prev
    x_tmp = diag_mul_rx(a_curr, x_prev, R)
    x_new = x_tmp + x_curr
    return a_new, x_new

@triton.autotune(configs=get_autotune_configs(), key=["L", "R"])
@triton.jit
def pscan_fwd_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch, stride_time,
    L, R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    A_IS_DIAG: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch

    r_idx = tl.arange(0, R)
    
    if A_IS_DIAG:
        a_ptrs = A_base + (offs[:, None] * stride_time) + r_idx[None, :]
        a_vals = tl.load(a_ptrs, mask=mask[:, None], other=1.0)
    else:
        r_row = r_idx[:, None]
        r_col = r_idx[None, :]
        a_ptrs = A_base + (offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
        a_vals = tl.load(a_ptrs, mask=mask[:, None, None], other=0.0)
        # Identity for padding
        eye = tl.zeros((R, R), dtype=tl.float32)
        for i in range(R): eye[i, i] = 1.0
        a_vals = tl.where(mask[:, None, None], a_vals, eye[None, :, :])

    x_row = r_idx[:, None]
    x_col = r_idx[None, :]
    x_ptrs = X_base + (offs[:, None, None] * stride_time) + (x_row[None, :, :] * R + x_col[None, :, :])
    x_vals = tl.load(x_ptrs, mask=mask[:, None, None], other=0.0)

    if A_IS_DIAG:
        acc_a, acc_x = tl.associative_scan((a_vals, x_vals), 0, combine_diag, R=R)
    else:
        acc_a, acc_x = tl.associative_scan((a_vals, x_vals), 0, combine_matrix, R=R)

    tl.store(Y_base + (offs[:, None, None] * stride_time) + (x_row[None, :, :] * R + x_col[None, :, :]), acc_x, mask=mask[:, None, None])

@triton.autotune(configs=get_autotune_configs(), key=["L", "R"])
@triton.jit
def pscan_bwd_kernel(
    A_ptr, X_grad_ptr, Y_grad_ptr, A_grad_ptr, Y_curr_ptr,
    stride_batch, stride_time,
    L, R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    A_IS_DIAG: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L

    # Pointers Setup
    offset_base = pid * stride_batch
    
    # Backward Scan Logic:
    # Reverse time scan. Inputs: X_grad (dL/dY). Multiplier: A^T (shifted by 1).
    # We scan to get accumulated gradients dY_acc.
    
    # Logic: reverse offs for loading
    read_offs = L - 1 - offs
    read_mask = read_offs >= 0 # Should match mask
    
    # Load A (need A[t+1] for step t in backward recurrence)
    # Shifted: at step t (reverse), we need A_{t+1}.
    # A_ptr points to A_0.
    # Load from read_offs + 1. If read_offs + 1 >= L, identity/zero.
    
    r_idx = tl.arange(0, R)
    r_row = r_idx[:, None]
    r_col = r_idx[None, :]
    
    a_read_offs = read_offs + 1
    a_mask = (a_read_offs < L) & mask

    if A_IS_DIAG:
        a_ptrs = A_ptr + offset_base + (a_read_offs[:, None] * stride_time) + r_idx[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask[:, None], other=0.0) 
        # For backward diag, transpose is same. Just identity for boundary.
        a_vals = tl.where(a_mask[:, None], a_vals, 1.0) # 1.0 is identity for mul
    else:
        # Load A and Transpose it immediately for Adjoint
        # A stored as [L, R, R]. We want A[t+1]^T.
        a_ptrs = A_ptr + offset_base + (a_read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
        a_vals_orig = tl.load(a_ptrs, mask=a_mask[:, None, None], other=0.0)
        # Transpose locally
        a_vals = tl.trans(a_vals_orig, 1, 2) # [BLOCK, R, R] -> Transpose last two dims? No, triton trans needs perm
        # Manual transpose in load or logic:
        # Actually easier to just load transposed indices: load (col * R + row)
        # But we loaded standard. Let's rebuild matrix logic.
        # matmul_r handles [BLOCK, R, R]. We need to transpose the (R, R) part.
        # Triton tl.trans is for 2D tensors. Here we have 3D (Block, R, R).
        # We perform manual transpose via value swapping or just logic change? 
        # Correct: use tl.permute? No.
        # Let's reload with transposed indices for simplicity.
        a_ptrs_T = A_ptr + offset_base + (a_read_offs[:, None, None] * stride_time) + (r_col[None, :, :] * R + r_row[None, :, :])
        a_vals = tl.load(a_ptrs_T, mask=a_mask[:, None, None], other=0.0)
        
        eye = tl.zeros((R, R), dtype=tl.float32)
        for i in range(R): eye[i, i] = 1.0
        a_vals = tl.where(a_mask[:, None, None], a_vals, eye[None, :, :])

    # Load X_grad (dL/dY)
    x_g_ptrs = X_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
    x_g_vals = tl.load(x_g_ptrs, mask=mask[:, None, None], other=0.0)

    # Perform Backward Scan
    if A_IS_DIAG:
        _, dY_acc = tl.associative_scan((a_vals, x_g_vals), 0, combine_diag, R=R)
    else:
        _, dY_acc = tl.associative_scan((a_vals, x_g_vals), 0, combine_matrix, R=R)

    # dY_acc is dX (dL/dX_t)
    # Store dX
    tl.store(X_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :]), dY_acc, mask=mask[:, None, None])

    # Compute dA
    # dA_t = dY_acc_t * Y_{t-1}^T
    # We need Y_{t-1}. Y pointer points to Y_0.
    # Load Y from read_offs - 1. If < 0, Zero.
    
    y_read_offs = read_offs - 1
    y_mask = (y_read_offs >= 0) & mask
    
    y_ptrs = Y_curr_ptr + offset_base + (y_read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
    y_vals = tl.load(y_ptrs, mask=y_mask[:, None, None], other=0.0)

    # dA calculation
    # For Diag: dA = dY * Y_prev (Elementwise sum over last dim? No, Diag A implies elementwise mul).
    # dL/dA_diag_t = dY_t * Y_{t-1}.
    if A_IS_DIAG:
        # dY_acc: [Block, R, R], Y_vals: [Block, R, R]
        # But A is [Block, R]. We need to sum over the second R dimension?
        # A_diag acts on X columns.
        # Y_t = A_t * Y_{t-1} + X. Here * is broadcasting (R,1) * (R,R).
        # dA_t = sum(dY_t * Y_{t-1}, axis=-1).
        da_val = tl.sum(dY_acc * y_vals, axis=2) # [Block, R]
        
        # Store dA
        da_ptrs = A_grad_ptr + offset_base + (read_offs[:, None] * stride_time) + r_idx[None, :]
        tl.store(da_ptrs, da_val, mask=mask[:, None])
    else:
        # Matrix Case: dA_t = dY_t @ Y_{t-1}^T
        # dY_acc: [Block, R, R]
        # Y_vals: [Block, R, R]
        # Result: [Block, R, R]
        # Manual MatMul again: C = A @ B^T
        # B^T indices: swap row/col
        
        da_mat = tl.zeros((R, R), dtype=tl.float32)
        # Loop over K (inner dim)
        for k in range(R):
            # dY [i, k] * Y [j, k] (since Y is transposed)
            da_mat += dY_acc[:, :, k, None] * y_vals[:, None, k, :]
            
        da_ptrs = A_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
        tl.store(da_ptrs, da_mat, mask=mask[:, None, None])

def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

class PScanTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        # A: (B, L, C, R) [Diag] or (B, L, C, R, R) [Matrix]
        # X: (B, L, C, R, R)
        ctx.save_for_backward(A, X)
        
        B, L, C, R1 = A.shape[:4]
        is_diag = (A.ndim == 4)
        
        if is_diag:
            R = R1
            # Flatten to (N, L, R)
            A_in = A.permute(0, 2, 1, 3).reshape(B*C, L, R).contiguous()
        else:
            R = R1
            R2 = A.shape[4]
            assert R == R2, "A must be square matrix"
            A_in = A.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()

        X_in = X.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        Y = torch.empty_like(X_in)
        
        BLOCK_SIZE = next_power_of_2(L)
        # Hardware limits check
        if BLOCK_SIZE < 16: BLOCK_SIZE = 16
        
        # Grid: (Batch * Channel)
        grid = (B * C, )
        
        pscan_fwd_kernel[grid](
            A_in, X_in, Y,
            A_in.stride(0), A_in.stride(1),
            L, R, BLOCK_SIZE,
            is_diag
        )
        
        # Reshape Y back
        Y_out = Y.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
        ctx.y_out = Y_out 
        ctx.is_diag = is_diag
        
        return Y_out

    @staticmethod
    def backward(ctx, grad_output):
        A, X = ctx.saved_tensors
        Y = ctx.y_out
        is_diag = ctx.is_diag
        
        B, L, C, R = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        
        # Prepare inputs for backward kernel
        # X_grad corresponds to incoming grad_output (dL/dY)
        X_grad = grad_output.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        
        # A input (Original A)
        if is_diag:
            A_in = A.permute(0, 2, 1, 3).reshape(B*C, L, R).contiguous()
            A_grad = torch.empty_like(A_in)
        else:
            A_in = A.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
            A_grad = torch.empty_like(A_in)
            
        Y_curr = Y.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        
        BLOCK_SIZE = next_power_of_2(L)
        if BLOCK_SIZE < 16: BLOCK_SIZE = 16
        grid = (B * C, )
        
        # In-place update X_grad to be dX
        # A_grad will be computed
        pscan_bwd_kernel[grid](
            A_in, X_grad, None, A_grad, Y_curr,
            A_in.stride(0), A_in.stride(1),
            L, R, BLOCK_SIZE,
            is_diag
        )
        
        # Reshape Grads
        dX = X_grad.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
        if is_diag:
            dA = A_grad.view(B, C, L, R).permute(0, 2, 1, 3)
        else:
            dA = A_grad.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
            
        return dA, dX

class PScan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A, X):
        return PScanTritonFunction.apply(A, X)
    