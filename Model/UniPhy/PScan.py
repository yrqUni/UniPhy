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
def scan_combine_matrix(a_prev, x_prev, a_curr, x_curr):
    R = a_prev.shape[1]
    a_new = tl.zeros_like(a_prev)
    x_new = tl.zeros_like(x_prev)
    for k in range(R):
        a_ik = a_curr[:, :, k, None]
        a_new += a_ik * a_prev[:, k, None, :]
        x_new += a_ik * x_prev[:, k, None, :]
    x_new += x_curr
    return a_new, x_new

@triton.jit
def scan_combine_diag(a_prev, x_prev, a_curr, x_curr):
    a_new = a_curr * a_prev
    x_new = a_curr * x_prev + x_curr
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
        a_vals_diag = tl.load(a_ptrs, mask=mask[:, None], other=1.0)
        a_vals = tl.broadcast_to(a_vals_diag[:, :, None], (BLOCK_SIZE, R, R))
    else:
        r_row = r_idx[:, None]
        r_col = r_idx[None, :]
        a_ptrs = A_base + (offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
        a_vals = tl.load(a_ptrs, mask=mask[:, None, None], other=0.0)
        eye = tl.zeros((R, R), dtype=tl.float32)
        for i in range(R): eye[i, i] = 1.0
        a_vals = tl.where(mask[:, None, None], a_vals, eye[None, :, :])

    x_row = r_idx[:, None]
    x_col = r_idx[None, :]
    x_ptrs = X_base + (offs[:, None, None] * stride_time) + (x_row[None, :, :] * R + x_col[None, :, :])
    x_vals = tl.load(x_ptrs, mask=mask[:, None, None], other=0.0)

    if A_IS_DIAG:
        acc_a, acc_x = tl.associative_scan((a_vals, x_vals), 0, scan_combine_diag)
    else:
        acc_a, acc_x = tl.associative_scan((a_vals, x_vals), 0, scan_combine_matrix)

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

    offset_base = pid * stride_batch
    
    read_offs = L - 1 - offs
    
    r_idx = tl.arange(0, R)
    r_row = r_idx[:, None]
    r_col = r_idx[None, :]
    
    a_read_offs = read_offs + 1
    a_mask = (a_read_offs < L) & mask

    if A_IS_DIAG:
        a_ptrs = A_ptr + offset_base + (a_read_offs[:, None] * stride_time) + r_idx[None, :]
        a_vals_diag = tl.load(a_ptrs, mask=a_mask[:, None], other=0.0) 
        a_vals_diag = tl.where(a_mask[:, None], a_vals_diag, 1.0)
        a_vals = tl.broadcast_to(a_vals_diag[:, :, None], (BLOCK_SIZE, R, R))
    else:
        a_ptrs_T = A_ptr + offset_base + (a_read_offs[:, None, None] * stride_time) + (r_col[None, :, :] * R + r_row[None, :, :])
        a_vals = tl.load(a_ptrs_T, mask=a_mask[:, None, None], other=0.0)
        eye = tl.zeros((R, R), dtype=tl.float32)
        for i in range(R): eye[i, i] = 1.0
        a_vals = tl.where(a_mask[:, None, None], a_vals, eye[None, :, :])

    x_g_ptrs = X_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
    x_g_vals = tl.load(x_g_ptrs, mask=mask[:, None, None], other=0.0)

    if A_IS_DIAG:
        _, dY_acc = tl.associative_scan((a_vals, x_g_vals), 0, scan_combine_diag)
    else:
        _, dY_acc = tl.associative_scan((a_vals, x_g_vals), 0, scan_combine_matrix)

    tl.store(X_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :]), dY_acc, mask=mask[:, None, None])

    y_read_offs = read_offs - 1
    y_mask = (y_read_offs >= 0) & mask
    
    y_ptrs = Y_curr_ptr + offset_base + (y_read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
    y_vals = tl.load(y_ptrs, mask=y_mask[:, None, None], other=0.0)

    if A_IS_DIAG:
        da_val = tl.sum(dY_acc * y_vals, axis=2)
        da_ptrs = A_grad_ptr + offset_base + (read_offs[:, None] * stride_time) + r_idx[None, :]
        tl.store(da_ptrs, da_val, mask=mask[:, None])
    else:
        da_mat = tl.zeros((R, R), dtype=tl.float32)
        for k in range(R):
            da_mat += dY_acc[:, :, k, None] * y_vals[:, None, k, :]
        da_ptrs = A_grad_ptr + offset_base + (read_offs[:, None, None] * stride_time) + (r_row[None, :, :] * R + r_col[None, :, :])
        tl.store(da_ptrs, da_mat, mask=mask[:, None, None])

def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

class PScanTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        B, L, C, R1 = A.shape[:4]
        is_diag = (A.ndim == 4)
        
        if is_diag:
            R = R1
            A_in = A.permute(0, 2, 1, 3).reshape(B*C, L, R).contiguous()
        else:
            R = R1
            A_in = A.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()

        X_in = X.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        Y = torch.empty_like(X_in)
        
        BLOCK_SIZE = next_power_of_2(L)
        if BLOCK_SIZE < 16: BLOCK_SIZE = 16
        
        grid = (B * C, )
        
        pscan_fwd_kernel[grid](
            A_in, X_in, Y,
            A_in.stride(0), A_in.stride(1),
            L, R, BLOCK_SIZE,
            is_diag
        )
        
        Y_out = Y.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
        ctx.save_for_backward(A, X, Y_out)
        ctx.is_diag = is_diag
        
        return Y_out

    @staticmethod
    def backward(ctx, grad_output):
        A, X, Y = ctx.saved_tensors
        is_diag = ctx.is_diag
        
        B, L, C, R = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        
        X_grad = grad_output.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        Y_curr = Y.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
        
        if is_diag:
            A_in = A.permute(0, 2, 1, 3).reshape(B*C, L, R).contiguous()
            A_grad = torch.empty_like(A_in)
        else:
            A_in = A.permute(0, 2, 1, 3, 4).reshape(B*C, L, R*R).contiguous()
            A_grad = torch.empty_like(A_in)
            
        BLOCK_SIZE = next_power_of_2(L)
        if BLOCK_SIZE < 16: BLOCK_SIZE = 16
        grid = (B * C, )
        
        pscan_bwd_kernel[grid](
            A_in, X_grad, None, A_grad, Y_curr,
            A_in.stride(0), A_in.stride(1),
            L, R, BLOCK_SIZE,
            is_diag
        )
        
        dX = X_grad.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
        if is_diag:
            dA = A_grad.view(B, C, L, R).permute(0, 2, 1, 3)
        else:
            dA = A_grad.view(B, C, L, R, R).permute(0, 2, 1, 3, 4)
            
        return dA, dX

class PScan(torch.nn.Module):
    def forward(self, A, X):
        return PScanTritonFunction.apply(A, X)
    