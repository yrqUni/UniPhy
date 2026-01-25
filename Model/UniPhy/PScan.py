import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def combine_diag(alr, ali, xlr, xli, arr, ari, xrr, xri):
    rar = arr * alr - ari * ali
    rai = arr * ali + ari * alr
    rxr = arr * xlr - ari * xli + xrr
    rxi = arr * xli + ari * xlr + xri
    return rar, rai, rxr, rxi

@triton.jit
def combine_mat(alr, ali, xlr, xli, arr, ari, xrr, xri):
    rar = tl.dot(arr, alr) - tl.dot(ari, ali)
    rai = tl.dot(arr, ali) + tl.dot(ari, alr)
    rxr = tl.dot(arr, xlr) - tl.dot(ari, xli) + xrr
    rxi = tl.dot(arr, xli) + tl.dot(ari, xlr) + xri
    return rar, rai, rxr, rxi

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["L"],
)
@triton.jit
def pscan_diag_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_b, stride_t,
    L, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    
    a_base = A_ptr + pid * stride_b + offs * stride_t
    x_base = X_ptr + pid * stride_b + offs * stride_t
    
    ar = tl.load(a_base, mask=mask, other=0.0)
    ai = tl.load(a_base + 1, mask=mask, other=0.0)
    xr = tl.load(x_base, mask=mask, other=0.0)
    xi = tl.load(x_base + 1, mask=mask, other=0.0)
    
    _, _, yr, yi = tl.associative_scan((ar, ai, xr, xi), axis=0, combine_fn=combine_diag)
    
    y_base = Y_ptr + pid * stride_b + offs * stride_t
    tl.store(y_base, yr, mask=mask)
    tl.store(y_base + 1, yi, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["L", "D"],
)
@triton.jit
def pscan_mat_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_ba, stride_ta, stride_ra, stride_ca,
    stride_bx, stride_tx, stride_dx,
    L, D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_t = tl.arange(0, BLOCK_SIZE)
    offs_r = tl.arange(0, D)
    offs_c = tl.arange(0, D)
    mask_t = offs_t < L

    a_ptrs = A_ptr + pid * stride_ba + offs_t[:, None, None] * stride_ta + \
             offs_r[None, :, None] * stride_ra + offs_c[None, None, :] * stride_ca
    
    ar = tl.load(a_ptrs, mask=mask_t[:, None, None], other=0.0)
    ai = tl.load(a_ptrs + 1, mask=mask_t[:, None, None], other=0.0)
    
    x_ptrs = X_ptr + pid * stride_bx + offs_t[:, None, None] * stride_tx + \
             offs_r[None, :, None] * stride_dx
             
    xr_vec = tl.load(x_ptrs, mask=mask_t[:, None, None], other=0.0)
    xi_vec = tl.load(x_ptrs + 1, mask=mask_t[:, None, None], other=0.0)
    
    xr = tl.where(offs_c[None, None, :] == 0, xr_vec, 0.0)
    xi = tl.where(offs_c[None, None, :] == 0, xi_vec, 0.0)
    
    _, _, yr_mat, yi_mat = tl.associative_scan((ar, ai, xr, xi), axis=0, combine_fn=combine_mat)
    
    y_ptrs = Y_ptr + pid * stride_bx + offs_t[:, None] * stride_tx + offs_r[None, :] * stride_dx
    tl.store(y_ptrs, yr_mat[:, :, 0], mask=mask_t[:, None])
    tl.store(y_ptrs + 1, yi_mat[:, :, 0], mask=mask_t[:, None])

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        is_mat = A.ndim == X.ndim + 1
        ctx.is_mat = is_mat
        B, L, D = X.shape
        ctx.save_for_backward(A, X)
        
        if is_mat:
            A_real = torch.view_as_real(A.contiguous())
            X_real = torch.view_as_real(X.contiguous())
            Y = torch.empty_like(X_real)
            bs = next_power_of_2(L)
            pscan_mat_kernel[(B,)](
                A_real, X_real, Y,
                A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                X_real.stride(0), X_real.stride(1), X_real.stride(2),
                L, D, BLOCK_SIZE=bs
            )
            res = torch.view_as_complex(Y)
            ctx.saved_Y = res
            return res
        else:
            A_flat = A.transpose(1, 2).reshape(-1, L).contiguous()
            X_flat = X.transpose(1, 2).reshape(-1, L).contiguous()
            A_real = torch.view_as_real(A_flat)
            X_real = torch.view_as_real(X_flat)
            Y = torch.empty_like(X_real)
            bs = next_power_of_2(L)
            pscan_diag_kernel[(A_flat.shape[0],)](
                A_real, X_real, Y,
                A_real.stride(0), A_real.stride(1),
                L, BLOCK_SIZE=bs
            )
            res = torch.view_as_complex(Y).view(B, D, L).transpose(1, 2)
            ctx.saved_Y = res
            return res

    @staticmethod
    def backward(ctx, grad_output):
        A, X = ctx.saved_tensors
        Y = ctx.saved_Y
        is_mat = ctx.is_mat
        B, L, D = X.shape
        g = grad_output.contiguous()
        
        if is_mat:
            A_H = A.conj().transpose(-1, -2)
            A_rev = torch.empty_like(A_H)
            A_rev[:, 0] = 0.0
            if L > 1:
                A_rev[:, 1:] = A_H[:, 1:].flip(1)
            X_rev = g.flip(1)
            
            A_real = torch.view_as_real(A_rev.contiguous())
            X_real = torch.view_as_real(X_rev.contiguous())
            Y_rev = torch.empty_like(X_real)
            bs = next_power_of_2(L)
            pscan_mat_kernel[(B,)](
                A_real, X_real, Y_rev,
                A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                X_real.stride(0), X_real.stride(1), X_real.stride(2),
                L, D, BLOCK_SIZE=bs
            )
            dX = torch.view_as_complex(Y_rev).flip(1)
            Y_prev = torch.zeros_like(Y)
            Y_prev[:, 1:] = Y[:, :-1]
            dA = dX.unsqueeze(-1) @ Y_prev.conj().unsqueeze(-2)
            return dA, dX
        else:
            A_conj = A.conj()
            A_rev = torch.empty_like(A_conj)
            A_rev[:, 0] = 0.0
            if L > 1:
                A_rev[:, 1:] = A_conj[:, 1:].flip(1)
            X_rev = g.flip(1)
            
            A_rev_flat = A_rev.transpose(1, 2).reshape(-1, L).contiguous()
            X_rev_flat = X_rev.transpose(1, 2).reshape(-1, L).contiguous()
            A_real = torch.view_as_real(A_rev_flat)
            X_real = torch.view_as_real(X_rev_flat)
            Y_rev = torch.empty_like(X_real)
            bs = next_power_of_2(L)
            pscan_diag_kernel[(A_rev_flat.shape[0],)](
                A_real, X_real, Y_rev,
                A_real.stride(0), A_real.stride(1),
                L, BLOCK_SIZE=bs
            )
            dX = torch.view_as_complex(Y_rev).view(B, D, L).transpose(1, 2).flip(1)
            Y_prev = torch.zeros_like(Y)
            Y_prev[:, 1:] = Y[:, :-1]
            dA = dX * Y_prev.conj()
            return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    