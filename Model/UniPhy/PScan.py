import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@triton.jit
def complex_matmul_op(ar, ai, br, bi):
    res_r = tl.dot(ar, br) - tl.dot(ai, bi)
    res_i = tl.dot(ar, bi) + tl.dot(ai, br)
    return res_r, res_i

@triton.jit
def scan_combine_diag(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar, new_ai = complex_mul(br, bi, ar, ai)
    bx_r, bx_i = complex_mul(br, bi, xr, xi)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

@triton.jit
def scan_combine_mat(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar, new_ai = complex_matmul_op(br, bi, ar, ai)
    bx_r, bx_i = tl.dot(br, xr) - tl.dot(bi, xi), tl.dot(br, xi) + tl.dot(bi, xr)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

def get_configs():
    return [
        triton.Config({}, num_warps=2),
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

    a_vals = tl.load(A_base + offs * stride_time, mask=mask, other=0.0)
    x_vals = tl.load(X_base + offs * stride_time, mask=mask, other=0.0)
    
    ar, ai = a_vals.real, a_vals.imag
    xr, xi = x_vals.real, x_vals.imag

    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan(
        (ar, ai, xr, xi), axis=0, combine_fn=scan_combine_diag
    )
    
    y_real = acc_xr
    y_imag = acc_xi
    tl.store(Y_base + offs * stride_time, y_real + 1j * y_imag, mask=mask)

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
    X_ptr_base = X_ptr + pid * stride_batch_x + offs[:, None] * stride_time_x
    Y_ptr_base = Y_ptr + pid * stride_batch_x + offs[:, None] * stride_time_x
    
    a_ptrs = A_ptr_base + row_offs[None, :, None] * stride_row_a + col_offs[None, None, :] * stride_col_a
    x_ptrs = X_ptr_base + row_offs[None, :] * stride_dim_x
    
    a_vals = tl.load(a_ptrs, mask=mask[:, None, None], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask[:, None], other=0.0)
    
    ar, ai = a_vals.real, a_vals.imag
    xr, xi = x_vals.real, x_vals.imag

    _, _, acc_xr, acc_xi = tl.associative_scan(
        (ar, ai, xr, xi), axis=0, combine_fn=scan_combine_mat
    )
    
    y_out = acc_xr + 1j * acc_xi
    tl.store(Y_ptr_base + row_offs[None, :] * stride_dim_x, y_out, mask=mask[:, None])

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
            A = A.contiguous()
            X = X.contiguous()
            Y = torch.empty_like(X)
            
            BLOCK_SIZE = next_power_of_2(L)
            if BLOCK_SIZE < 16: BLOCK_SIZE = 16
            
            grid = (B,)
            pscan_mat_kernel[grid](
                A, X, Y,
                A.stride(0), A.stride(1), A.stride(2), A.stride(3),
                X.stride(0), X.stride(1), X.stride(2),
                L, D, BLOCK_SIZE=BLOCK_SIZE
            )
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
            Y_flat = torch.empty_like(X_flat)
            
            BLOCK_SIZE = next_power_of_2(L)
            if BLOCK_SIZE < 16: BLOCK_SIZE = 16
            
            grid = (A_flat.shape[0],)
            pscan_diag_kernel[grid](
                A_flat, X_flat, Y_flat,
                A_flat.stride(0), A_flat.stride(1),
                L, BLOCK_SIZE=BLOCK_SIZE
            )
            
            Y = Y_flat.view(B, D, L).transpose(1, 2)
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
            
            A_conj = A.conj().permute(0, 1, 3, 2)
            A_rev = torch.cat([torch.zeros_like(A_conj[:, 0:1]), A_conj[:, :-1]], dim=1)
            A_rev = torch.flip(A_rev, [1]).contiguous()
            
            X_rev = torch.flip(grad_output, [1]).contiguous()
            dX_rev = torch.empty_like(X_rev)
            
            BLOCK_SIZE = next_power_of_2(L)
            if BLOCK_SIZE < 16: BLOCK_SIZE = 16
            grid = (B,)
            
            pscan_mat_kernel[grid](
                A_rev, X_rev, dX_rev,
                A_rev.stride(0), A_rev.stride(1), A_rev.stride(2), A_rev.stride(3),
                X_rev.stride(0), X_rev.stride(1), X_rev.stride(2),
                L, D, BLOCK_SIZE=BLOCK_SIZE
            )
            
            dX = torch.flip(dX_rev, [1])
            
            Y_shift = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
            dA = dX.unsqueeze(-1) @ Y_shift.conj().unsqueeze(-2)
            
            return dA, dX
            
        else:
            A = A.contiguous()
            if A.ndim == grad_output.ndim - 1:
                A = A.unsqueeze(-1)
            if A.shape != grad_output.shape:
                A, _ = torch.broadcast_tensors(A, grad_output)

            B, L, D = A.shape
            A_conj = A.conj()
            A_rev = torch.cat([torch.zeros_like(A_conj[:, 0:1]), A_conj[:, :-1]], dim=1)
            A_rev_flat = torch.flip(A_rev, [1]).transpose(1, 2).reshape(-1, L).contiguous()
            
            X_rev_flat = torch.flip(grad_output, [1]).transpose(1, 2).reshape(-1, L).contiguous()
            dX_rev_flat = torch.empty_like(X_rev_flat)
            
            BLOCK_SIZE = next_power_of_2(L)
            if BLOCK_SIZE < 16: BLOCK_SIZE = 16
            grid = (A_rev_flat.shape[0],)
            
            pscan_diag_kernel[grid](
                A_rev_flat, X_rev_flat, dX_rev_flat,
                A_rev_flat.stride(0), A_rev_flat.stride(1),
                L, BLOCK_SIZE=BLOCK_SIZE
            )
            
            dX_rev = dX_rev_flat.view(B, D, L).transpose(1, 2)
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
    