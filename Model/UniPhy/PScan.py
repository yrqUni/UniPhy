import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import triton
import triton.language as tl
import math

@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@triton.jit
def scan_combine(ar, ai, xr, xi, br, bi, yr, yi):
    new_ar, new_ai = complex_mul(br, bi, ar, ai)
    bx_r, bx_i = complex_mul(br, bi, xr, xi)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

def get_autotune_configs():
    return [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['L'],
)

@triton.jit
def pscan_kernel(
    A_ptr, X_ptr, Y_ptr, 
    stride_batch: tl.constexpr, stride_time: tl.constexpr, 
    L, BLOCK_SIZE: tl.constexpr, REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    
    read_offs = tl.where(REVERSE, (L - 1 - offs), offs)
    
    a_r = tl.load(A_base + read_offs * stride_time + 0, mask=mask, other=1.0)
    a_i = tl.load(A_base + read_offs * stride_time + 1, mask=mask, other=0.0)
    x_r = tl.load(X_base + read_offs * stride_time + 0, mask=mask, other=0.0)
    x_i = tl.load(X_base + read_offs * stride_time + 1, mask=mask, other=0.0)
    
    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan((a_r, a_i, x_r, x_i), axis=0, combine_fn=scan_combine)
    
    tl.store(Y_base + read_offs * stride_time + 0, acc_xr, mask=mask)
    tl.store(Y_base + read_offs * stride_time + 1, acc_xi, mask=mask)

def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        ctx.shape_A_orig = A.shape
        ctx.shape_X_orig = X.shape
        
        if A.ndim == X.ndim - 1: A = A.unsqueeze(-1)
        if A.shape != X.shape: A, X = torch.broadcast_tensors(A, X)
        
        A, X = A.contiguous(), X.contiguous()
        L = X.shape[1]
        
        A_in, X_in = A.transpose(1, -1).contiguous(), X.transpose(1, -1).contiguous()
        A_flat, X_flat = A_in.view(-1, L), X_in.view(-1, L)
        
        A_real, X_real = torch.view_as_real(A_flat).contiguous(), torch.view_as_real(X_flat).contiguous()
        Y_real = torch.empty_like(X_real)
        
        BLOCK_SIZE = max(16, next_power_of_2(L))
        
        pscan_kernel[(A_flat.shape[0],)](
            A_real, X_real, Y_real, 
            A_real.stride(0), A_real.stride(1), 
            L, BLOCK_SIZE, False
        )
        
        Y = torch.view_as_complex(Y_real).view(*A_in.shape).transpose(1, -1)
        ctx.save_for_backward(A, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, Y = ctx.saved_tensors
        A_conj = A.conj()
        A_prep = torch.cat([A_conj[:, 1:], torch.zeros_like(A_conj[:, 0:1])], dim=1)
        
        L = A.shape[1]
        A_in, X_in = A_prep.transpose(1, -1).contiguous(), grad_output.transpose(1, -1).contiguous()
        A_flat, X_flat = A_in.view(-1, L), X_in.view(-1, L)
        
        A_real, X_real = torch.view_as_real(A_flat).contiguous(), torch.view_as_real(X_flat).contiguous()
        Y_real = torch.empty_like(X_real)
        
        BLOCK_SIZE = max(16, next_power_of_2(L))
        
        pscan_kernel[(A_flat.shape[0],)](
            A_real, X_real, Y_real, 
            A_real.stride(0), A_real.stride(1), 
            L, BLOCK_SIZE, True
        )
        
        dX = torch.view_as_complex(Y_real).view(*A_in.shape).transpose(1, -1)
        
        Y_prev = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
        dA = dX * Y_prev.conj()
        
        for i, dim in enumerate(ctx.shape_A_orig):
            if dim == 1: dA = dA.sum(dim=i, keepdim=True)
            
        if dX.shape != ctx.shape_X_orig:
            for i, dim in enumerate(ctx.shape_X_orig):
                if dim == 1: dX = dX.sum(dim=i, keepdim=True)
                
        return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)

