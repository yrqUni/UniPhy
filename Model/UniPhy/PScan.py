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
def combine_mat(alr, ali, xlr, xli, arr, ari, xrr, xri, D: tl.constexpr):
    Alr = tl.reshape(alr, (D, D))
    Ali = tl.reshape(ali, (D, D))
    Arr = tl.reshape(arr, (D, D))
    Ari = tl.reshape(ari, (D, D))
    Xlr = tl.reshape(xlr, (D, D))
    Xli = tl.reshape(xli, (D, D))
    Xrr = tl.reshape(xrr, (D, D))
    Xri = tl.reshape(xri, (D, D))

    rar = tl.dot(Arr, Alr) - tl.dot(Ari, Ali)
    rai = tl.dot(Arr, Ali) + tl.dot(Ari, Alr)

    rxr = tl.dot(Arr, Xlr) - tl.dot(Ari, Xli) + Xrr
    rxi = tl.dot(Arr, Xli) + tl.dot(Ari, Xlr) + xri

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
    L, D: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_t = tl.arange(0, BLOCK_SIZE)
    offs_d2 = tl.arange(0, D * D)
    mask_t = offs_t < L

    a_base = A_ptr + pid * stride_ba + offs_t[:, None] * stride_ta + offs_d2[None, :] * 2
    x_base = X_ptr + pid * stride_bx + offs_t[:, None] * stride_tx + offs_d2[None, :] * 2
    
    ar = tl.load(a_base, mask=mask_t[:, None], other=0.0)
    ai = tl.load(a_base + 1, mask=mask_t[:, None], other=0.0)
    xr = tl.load(x_base, mask=mask_t[:, None], other=0.0)
    xi = tl.load(x_base + 1, mask=mask_t[:, None], other=0.0)

    def combine_fn(alr, ali, xlr, xli, arr, ari, xrr, xri):
        return combine_mat(alr, ali, xlr, xli, arr, ari, xrr, xri, D)

    _, _, yr_f, yi_f = tl.associative_scan((ar, ai, xr, xi), axis=0, combine_fn=combine_fn)
    
    y_out_base = Y_ptr + pid * stride_bx + offs_t[:, None] * stride_tx + tl.arange(0, D)[None, :] * 2
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
            A_real = torch.view_as_real(A.contiguous())
            X_pad = torch.zeros(B, L, D, D, dtype=X.dtype, device=X.device)
            X_pad[..., 0] = X
            X_real = torch.view_as_real(X_pad)
            Y = torch.empty_like(X_real)
            pscan_mat_kernel[(B,)](
                A_real, X_real, Y,
                A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                X_real.stride(0), X_real.stride(1), 2,
                L, D, BLOCK_SIZE=next_power_of_2(L)
            )
            res = torch.view_as_complex(Y)[..., 0]
            ctx.saved_Y = res
            return res
        else:
            A_flat = torch.view_as_real(A.transpose(1, 2).reshape(-1, L).contiguous())
            X_flat = torch.view_as_real(X.transpose(1, 2).reshape(-1, L).contiguous())
            Y = torch.empty_like(X_flat)
            pscan_diag_kernel[(A_flat.shape[0],)](
                A_flat, X_flat, Y,
                A_flat.stride(0), A_flat.stride(1),
                L, BLOCK_SIZE=next_power_of_2(L)
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
            A_rev = torch.eye(D, device=A.device, dtype=A.dtype).view(1, 1, D, D).repeat(B, L, 1, 1)
            A_rev[:, 0] = 0.0
            if L > 1: A_rev[:, 1:] = A_H[:, 1:].flip(1)
            X_rev = g.flip(1)
            A_real, X_pad = torch.view_as_real(A_rev.contiguous()), torch.zeros(B, L, D, D, dtype=X.dtype, device=X.device)
            X_pad[..., 0] = X_rev
            X_real, Y_rev = torch.view_as_real(X_pad), torch.empty_like(torch.view_as_real(X_pad))
            pscan_mat_kernel[(B,)](A_real, X_real, Y_rev, A_real.stride(0), A_real.stride(1), A_real.stride(2), A_real.stride(3),
                                   X_real.stride(0), X_real.stride(1), 2, L, D, BLOCK_SIZE=next_power_of_2(L))
            dX = torch.view_as_complex(Y_rev)[..., 0].flip(1)
            Y_prev = torch.zeros_like(Y)
            Y_prev[:, 1:] = Y[:, :-1]
            dA = dX.unsqueeze(-1) @ Y_prev.conj().unsqueeze(-2)
            return dA, dX
        else:
            A_rev = torch.zeros_like(A)
            if L > 1: A_rev[:, 1:] = A.conj()[:, 1:].flip(1)
            X_rev = g.flip(1)
            A_flat = torch.view_as_real(A_rev.transpose(1, 2).reshape(-1, L).contiguous())
            X_flat = torch.view_as_real(X_rev.transpose(1, 2).reshape(-1, L).contiguous())
            Y_rev = torch.empty_like(X_flat)
            pscan_diag_kernel[(A_flat.shape[0],)](A_flat, X_flat, Y_rev, A_flat.stride(0), A_flat.stride(1), L, BLOCK_SIZE=next_power_of_2(L))
            dX = torch.view_as_complex(Y_rev).view(B, D, L).transpose(1, 2).flip(1)
            Y_prev = torch.zeros_like(Y)
            Y_prev[:, 1:] = Y[:, :-1]
            dA = dX * Y_prev.conj()
            return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    