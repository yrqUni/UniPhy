import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_complex_manual(ar, ai, br, bi, D: tl.constexpr):
    ar_e = tl.reshape(ar, (D, D, 1))
    ai_e = tl.reshape(ai, (D, D, 1))
    br_e = tl.reshape(br, (1, D, D))
    bi_e = tl.reshape(bi, (1, D, D))
    
    rr = tl.sum(ar_e * br_e, axis=1)
    ri = tl.sum(ar_e * bi_e, axis=1)
    ir = tl.sum(ai_e * br_e, axis=1)
    ii = tl.sum(ai_e * bi_e, axis=1)
    
    return rr - ii, ri + ir

@triton.jit
def combine_diag(alr, ali, xlr, xli, arr, ari, xrr, xri):
    rar = arr * alr - ari * ali
    rai = arr * ali + ari * alr
    rxr = arr * xlr - ari * xli + xrr
    rxi = arr * xli + ari * xlr + xri
    return rar, rai, rxr, rxi

@triton.jit
def combine_mat(alr, ali, xlr, xli, arr, ari, xrr, xri, D: tl.constexpr):
    rar, rai = matmul_complex_manual(arr, ari, alr, ali, D)
    rxr_m, rxi_m = matmul_complex_manual(arr, ari, xlr, xli, D)
    rxr = rxr_m + xrr
    rxi = rxi_m + xri
    return tl.reshape(rar, (D * D,)), tl.reshape(rai, (D * D,)), \
           tl.reshape(rxr, (D * D,)), tl.reshape(rxi, (D * D,))

@triton.autotune(
    configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
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
    configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
    key=["L", "D"],
)
@triton.jit
def pscan_mat_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_ba, stride_ta, stride_ra, stride_ca,
    stride_bx, stride_tx, stride_dx,
    L, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, REVERSE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_t = tl.arange(0, BLOCK_SIZE)
    offs_d2 = tl.arange(0, D * D)
    mask_t = offs_t < L
    
    t_idx = tl.where(REVERSE, L - 1 - offs_t, offs_t)
    
    a_base = A_ptr + pid * stride_ba + t_idx[:, None] * stride_ta + offs_d2[None, :] * 2
    x_base = X_ptr + pid * stride_bx + t_idx[:, None] * stride_tx + tl.arange(0, D)[None, :] * stride_dx
    
    ar_f = tl.load(a_base, mask=mask_t[:, None], other=0.0)
    ai_f = tl.load(a_base + 1, mask=mask_t[:, None], other=0.0)
    
    xr_v = tl.load(x_base, mask=mask_t[:, None], other=0.0)
    xi_v = tl.load(x_base + 1, mask=mask_t[:, None], other=0.0)
    
    xr_mat = tl.reshape(xr_v, (BLOCK_SIZE, D, 1))
    xi_mat = tl.reshape(xi_v, (BLOCK_SIZE, D, 1))
    
    offs_c = tl.arange(0, D)
    xr_f = tl.reshape(tl.where(offs_c[None, None, :] == 0, xr_mat, 0.0), (BLOCK_SIZE, D * D))
    xi_f = tl.reshape(tl.where(offs_c[None, None, :] == 0, xi_mat, 0.0), (BLOCK_SIZE, D * D))

    def combine_fn(alr, ali, xlr, xli, arr, ari, xrr, xri):
        return combine_mat(alr, ali, xlr, xli, arr, ari, xrr, xri, D)

    _, _, yr_f, yi_f = tl.associative_scan((ar_f, ai_f, xr_f, xi_f), axis=0, combine_fn=combine_fn)
    
    y_out_base = Y_ptr + pid * stride_bx + t_idx[:, None] * stride_tx + tl.arange(0, D)[None, :] * stride_dx
    yr_mat = tl.reshape(yr_f, (BLOCK_SIZE, D, D))
    yi_mat = tl.reshape(yi_f, (BLOCK_SIZE, D, D))
    
    tl.store(y_out_base, yr_mat[:, :, 0], mask=mask_t[:, None])
    tl.store(y_out_base + 1, yi_mat[:, :, 0], mask=mask_t[:, None])

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
            Ar, Xr = torch.view_as_real(A.contiguous()), torch.view_as_real(X.contiguous())
            Y = torch.empty_like(Xr)
            pscan_mat_kernel[(B,)](Ar, Xr, Y, Ar.stride(0), Ar.stride(1), Ar.stride(2), Ar.stride(3),
                                   Xr.stride(0), Xr.stride(1), Xr.stride(2), L, D, next_power_of_2(L), False)
            res = torch.view_as_complex(Y)
            ctx.saved_Y = res
            return res
        else:
            Af = torch.view_as_real(A.transpose(1, 2).reshape(-1, L).contiguous())
            Xf = torch.view_as_real(X.transpose(1, 2).reshape(-1, L).contiguous())
            Y = torch.empty_like(Xf)
            pscan_diag_kernel[(Af.shape[0],)](Af, Xf, Y, Af.stride(0), Af.stride(1), L, next_power_of_2(L))
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
            Ah = A.conj().transpose(-1, -2)
            Ar = torch.zeros_like(Ah)
            if L > 1: Ar[:, 1:] = Ah[:, 1:]
            Ar_r, Gr_r = torch.view_as_real(Ar.contiguous()), torch.view_as_real(g.contiguous())
            Y_rev = torch.empty_like(Gr_r)
            pscan_mat_kernel[(B,)](Ar_r, Gr_r, Y_rev, Ar_r.stride(0), Ar_r.stride(1), Ar_r.stride(2), Ar_r.stride(3),
                                   Gr_r.stride(0), Gr_r.stride(1), Gr_r.stride(2), L, D, next_power_of_2(L), True)
            dX = torch.view_as_complex(Y_rev)
            Yp = torch.zeros_like(Y)
            Yp[:, 1:] = Y[:, :-1]
            dA = dX.unsqueeze(-1) @ Yp.conj().unsqueeze(-2)
            return dA, dX
        else:
            Ac = A.conj()
            Ar = torch.zeros_like(Ac)
            if L > 1: Ar[:, 1:] = Ac[:, 1:]
            Af = torch.view_as_real(Ar.transpose(1, 2).reshape(-1, L).contiguous())
            Gf = torch.view_as_real(g.transpose(1, 2).reshape(-1, L).contiguous())
            Y_rev = torch.empty_like(Gf)
            pscan_diag_kernel[(Af.shape[0],)](Af, Gf, Y_rev, Af.stride(0), Af.stride(1), L, next_power_of_2(L))
            dX = torch.view_as_complex(Y_rev).view(B, D, L).transpose(1, 2)
            Yp = torch.zeros_like(Y)
            Yp[:, 1:] = Y[:, :-1]
            dA = dX * Yp.conj()
            return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    