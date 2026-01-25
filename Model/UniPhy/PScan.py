import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batch_matmul_real(ar, ai, br, bi):
    # ar, ai: [BLOCK, D, D] (A)
    # br, bi: [BLOCK, D, D] (B)
    # Target: C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
    
    # Expand A to [BLOCK, D, D, 1] -> (b, i, k, 1)
    ar_ = tl.expand_dims(ar, 3)
    ai_ = tl.expand_dims(ai, 3)
    
    # Expand B to [BLOCK, 1, D, D] -> (b, 1, k, j)
    br_ = tl.expand_dims(br, 1)
    bi_ = tl.expand_dims(bi, 1)
    
    # Multiply: (b, i, k, 1) * (b, 1, k, j) -> (b, i, k, j)
    rr = ar_ * br_
    ri = ar_ * bi_
    ir = ai_ * br_
    ii = ai_ * bi_
    
    # Sum over k (axis 2)
    cr = tl.sum(rr - ii, axis=2)
    ci = tl.sum(ri + ir, axis=2)
    return cr, ci

@triton.jit
def scan_combine_diag(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar = ar * br - ai * bi
    new_ai = ar * bi + ai * br
    bx_r = br * xr - bi * xi
    bx_i = br * xi + bi * xr
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

@triton.jit
def scan_combine_mat(ar, ai, xr, xi, br, bi, yr, yi):
    # Matrix multiplication for A_new = B @ A
    new_ar, new_ai = batch_matmul_real(br, bi, ar, ai)
    
    # Matrix multiplication for X_new = B @ X + Y
    # Note: X and Y are loaded as [BLOCK, D, D] (broadcasted vectors)
    # So B @ X implies applying B to every column of X identicaly.
    bx_r, bx_i = batch_matmul_real(br, bi, xr, xi)
    
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

def get_configs():
    return [
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

@triton.autotune(configs=get_configs(), key=["L"])
@triton.jit
def pscan_diag_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch, stride_time,
    L, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    off_time = offs * stride_time
    
    ar = tl.load(A_base + off_time, mask=mask, other=0.0)
    ai = tl.load(A_base + off_time + 1, mask=mask, other=0.0)
    xr = tl.load(X_base + off_time, mask=mask, other=0.0)
    xi = tl.load(X_base + off_time + 1, mask=mask, other=0.0)

    _, _, acc_xr, acc_xi = tl.associative_scan(
        (ar, ai, xr, xi), axis=0, combine_fn=scan_combine_diag
    )
    
    tl.store(Y_base + off_time, acc_xr, mask=mask)
    tl.store(Y_base + off_time + 1, acc_xi, mask=mask)

@triton.autotune(configs=get_configs(), key=["L", "D"])
@triton.jit
def pscan_mat_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch_a, stride_time_a, stride_row_a, stride_col_a,
    stride_batch_x, stride_time_x, stride_dim_x,
    L, D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    offs = tl.arange(0, BLOCK_SIZE)
    row_offs = tl.arange(0, D)
    col_offs = tl.arange(0, D)
    
    mask = offs < L
    
    A_ptr_base = A_ptr + pid * stride_batch_a + offs[:, None, None] * stride_time_a
    X_ptr_base = X_ptr + pid * stride_batch_x + offs[:, None, None] * stride_time_x
    Y_ptr_base = Y_ptr + pid * stride_batch_x + offs[:, None] * stride_time_x
    
    a_ptrs = A_ptr_base + row_offs[None, :, None] * stride_row_a + col_offs[None, None, :] * stride_col_a
    
    # Broadcast X vector to Matrix shape [BLOCK, D, D]
    # stride_col is 0, so every column gets the same vector value
    x_ptrs = X_ptr_base + row_offs[None, :, None] * stride_dim_x + col_offs[None, None, :] * 0
    
    ar = tl.load(a_ptrs, mask=mask[:, None, None], other=0.0)
    ai = tl.load(a_ptrs + 1, mask=mask[:, None, None], other=0.0)
    xr = tl.load(x_ptrs, mask=mask[:, None, None], other=0.0)
    xi = tl.load(x_ptrs + 1, mask=mask[:, None, None], other=0.0)

    _, _, acc_xr, acc_xi = tl.associative_scan(
        (ar, ai, xr, xi), axis=0, combine_fn=scan_combine_mat
    )
    
    # Output is technically [BLOCK, D, D] (replicated columns), take column 0
    final_xr = acc_xr[:, :, 0]
    final_xi = acc_xi[:, :, 0]
    
    y_ptrs = Y_ptr_base + row_offs[None, :] * stride_dim_x
    tl.store(y_ptrs, final_xr, mask=mask[:, None])
    tl.store(y_ptrs + 1, final_xi, mask=mask[:, None])

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        ctx.save_for_backward(A, X)
        is_matrix = A.ndim == X.ndim + 1
        ctx.is_matrix = is_matrix
        
        if is_matrix:
            B, L, D, _ = A.shape
            A_real = torch.view_as_real(A).contiguous()
            X_real = torch.view_as_real(X).contiguous()
            Y_real = torch.empty_like(X_real)
            
            BLOCK_SIZE = max(16, next_power_of_2(L))
            grid = (B,)
            
            pscan_mat_kernel[grid](
                A_real, X_real, Y_real,
                A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                X_real.stride(0), X_real.stride(1), X_real.stride(2),
                L, D, BLOCK_SIZE=BLOCK_SIZE
            )
            Y = torch.view_as_complex(Y_real)
            ctx.saved_Y = Y
            return Y
        else:
            if A.ndim == X.ndim - 1:
                A = A.unsqueeze(-1)
            if A.shape != X.shape:
                A, X = torch.broadcast_tensors(A, X)
            
            A = A.contiguous()
            X = X.contiguous()
            B, L, D = A.shape
            
            A_flat = A.transpose(1, 2).reshape(-1, L).contiguous()
            X_flat = X.transpose(1, 2).reshape(-1, L).contiguous()
            
            A_real = torch.view_as_real(A_flat)
            X_real = torch.view_as_real(X_flat)
            Y_real = torch.empty_like(X_real)
            
            BLOCK_SIZE = max(16, next_power_of_2(L))
            grid = (A_flat.shape[0],)
            
            pscan_diag_kernel[grid](
                A_real, X_real, Y_real,
                A_real.stride(0), A_real.stride(1),
                L, BLOCK_SIZE=BLOCK_SIZE
            )
            
            Y = torch.view_as_complex(Y_real).view(B, D, L).transpose(1, 2)
            ctx.saved_Y = Y
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, _ = ctx.saved_tensors
        Y = ctx.saved_Y
        is_matrix = ctx.is_matrix
        grad_output = grad_output.contiguous()
        
        if is_matrix:
            B, L, D, _ = A.shape
            
            A_H = A.conj().permute(0, 1, 3, 2)
            A_shifted = A_H[:, 1:]
            A_rev_part = torch.flip(A_shifted, [1])
            A_rev = torch.cat([torch.zeros_like(A_H[:, 0:1]), A_rev_part], dim=1).contiguous()
            
            X_rev = torch.flip(grad_output, [1]).contiguous()
            
            A_real = torch.view_as_real(A_rev)
            X_real = torch.view_as_real(X_rev)
            dX_rev_real = torch.empty_like(X_real)
            
            BLOCK_SIZE = max(16, next_power_of_2(L))
            grid = (B,)
            
            pscan_mat_kernel[grid](
                A_real, X_real, dX_rev_real,
                A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                X_real.stride(0), X_real.stride(1), X_real.stride(2),
                L, D, BLOCK_SIZE=BLOCK_SIZE
            )
            
            dX = torch.flip(torch.view_as_complex(dX_rev_real), [1])
            
            Y_shift = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
            dA = dX.unsqueeze(-1) @ Y_shift.conj().unsqueeze(-2)
            
            return dA, dX
            
        else:
            if A.ndim == grad_output.ndim - 1:
                A = A.unsqueeze(-1)
            if A.shape != grad_output.shape:
                A, _ = torch.broadcast_tensors(A, grad_output)

            B, L, D = A.shape
            
            A_conj = A.conj()
            A_shifted = A_conj[:, 1:]
            A_rev_part = torch.flip(A_shifted, [1])
            A_rev = torch.cat([torch.zeros_like(A_conj[:, 0:1]), A_rev_part], dim=1).contiguous()
            
            A_rev_flat = A_rev.transpose(1, 2).reshape(-1, L).contiguous()
            X_rev_flat = torch.flip(grad_output, [1]).transpose(1, 2).reshape(-1, L).contiguous()
            
            A_real = torch.view_as_real(A_rev_flat)
            X_real = torch.view_as_real(X_rev_flat)
            dX_rev_real = torch.empty_like(X_real)
            
            BLOCK_SIZE = max(16, next_power_of_2(L))
            grid = (A_rev_flat.shape[0],)
            
            pscan_diag_kernel[grid](
                A_real, X_real, dX_rev_real,
                A_real.stride(0), A_real.stride(1),
                L, BLOCK_SIZE=BLOCK_SIZE
            )
            
            dX_rev = torch.view_as_complex(dX_rev_real).view(B, D, L).transpose(1, 2)
            dX = torch.flip(dX_rev, [1])
            
            Y_shift = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
            dA = dX * Y_shift.conj()
            
            if dA.shape != ctx.saved_tensors[0].shape:
                red_dims = []
                for i, (in_dim, out_dim) in enumerate(zip(ctx.saved_tensors[0].shape, dA.shape)):
                    if in_dim == 1 and out_dim > 1:
                        red_dims.append(i)
                if red_dims:
                    dA = dA.sum(dim=red_dims, keepdim=True)
            
            return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    