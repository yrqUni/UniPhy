import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def complex_mul_mat_mat(ar, ai, br, bi, R: tl.constexpr):
    ar = tl.reshape(ar, (R, R))
    ai = tl.reshape(ai, (R, R))
    br = tl.reshape(br, (R, R))
    bi = tl.reshape(bi, (R, R))
    out_r = tl.dot(br, ar) - tl.dot(bi, ai)
    out_i = tl.dot(br, ai) + tl.dot(bi, ar)
    return tl.ravel(out_r), tl.ravel(out_i)

@triton.jit
def complex_mul_mat_vec(br, bi, xr, xi, R: tl.constexpr):
    br = tl.reshape(br, (R, R))
    bi = tl.reshape(bi, (R, R))
    xr = tl.reshape(xr, (R, 1))
    xi = tl.reshape(xi, (R, 1))
    out_r = tl.dot(br, xr) - tl.dot(bi, xi)
    out_i = tl.dot(br, xi) + tl.dot(bi, xr)
    return tl.ravel(out_r), tl.ravel(out_i)

@triton.jit
def combine_fn(ar, ai, xr, xi, br, bi, yr, yi, R: tl.constexpr):
    nar, nai = complex_mul_mat_mat(ar, ai, br, bi, R)
    bx_r, bx_i = complex_mul_mat_vec(br, bi, xr, xi, R)
    nxr = bx_r + yr
    nxi = bx_i + yi
    return nar, nai, nxr, nxi

@triton.jit
def pscan_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_ab, stride_al, stride_ac,
    stride_xb, stride_xl, stride_xc,
    L, R: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    
    offs_l = tl.arange(0, BLOCK_SIZE)
    mask_l = offs_l < L
    idx_l = (L - 1 - offs_l) if REVERSE else offs_l

    offs_mat = tl.arange(0, R * R)
    offs_vec = tl.arange(0, R)

    a_base = A_ptr + pid_b * stride_ab + pid_c * stride_ac
    x_base = X_ptr + pid_b * stride_xb + pid_c * stride_xc
    
    a_r = tl.load(a_base + idx_l[:, None] * stride_al + offs_mat[None, :], mask=mask_l[:, None], other=0.0)
    a_i = tl.load(a_base + idx_l[:, None] * stride_al + offs_mat[None, :] + (R*R), mask=mask_l[:, None], other=0.0)
    
    x_r = tl.load(x_base + idx_l[:, None] * stride_xl + offs_vec[None, :], mask=mask_l[:, None], other=0.0)
    x_i = tl.load(x_base + idx_l[:, None] * stride_xl + offs_vec[None, :] + R, mask=mask_l[:, None], other=0.0)

    if REVERSE:
        eye_mask = (offs_mat // R) == (offs_mat % R)
        a_r = tl.where(offs_l[:, None] == 0, tl.where(eye_mask, 1.0, 0.0), a_r)
        a_i = tl.where(offs_l[:, None] == 0, 0.0, a_i)

    res_ar, res_ai, res_xr, res_xi = tl.associative_scan(
        (a_r, a_i, x_r, x_i), axis=0, 
        combine_fn=combine_fn
    )

    y_base = Y_ptr + pid_b * stride_xb + pid_c * stride_xc
    tl.store(y_base + idx_l[:, None] * stride_xl + offs_vec[None, :], res_xr, mask=mask_l[:, None])
    tl.store(y_base + idx_l[:, None] * stride_xl + offs_vec[None, :] + R, res_xi, mask=mask_l[:, None])

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        ctx.is_diag = A.ndim == 4
        if ctx.is_diag:
            A = torch.diag_embed(A)
        
        B, L, C, R, _ = A.shape
        A, X = A.contiguous(), X.contiguous()
        Y = torch.empty_like(X)
        BLOCK_SIZE = triton.next_power_of_2(L)
        
        pscan_kernel[(B, C)](
            torch.view_as_real(A), torch.view_as_real(X), torch.view_as_real(Y),
            A.stride(0), A.stride(1), A.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            L, R, BLOCK_SIZE, False,
            num_warps=8 if R >= 16 else 4
        )
        ctx.save_for_backward(A, X, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, X, Y = ctx.saved_tensors
        B, L, C, R, _ = A.shape
        
        A_conj_T = A.conj().transpose(-1, -2).contiguous()
        A_prep = torch.zeros_like(A_conj_T)
        A_prep[:, :-1] = A_conj_T[:, 1:]
        A_prep[:, -1] = torch.eye(R, device=A.device, dtype=A.dtype).expand(B, C, R, R)

        grad_output = grad_output.contiguous()
        dX = torch.empty_like(grad_output)
        BLOCK_SIZE = triton.next_power_of_2(L)
        
        pscan_kernel[(B, C)](
            torch.view_as_real(A_prep), torch.view_as_real(grad_output), torch.view_as_real(dX),
            A_prep.stride(0), A_prep.stride(1), A_prep.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L, R, BLOCK_SIZE, True,
            num_warps=8 if R >= 16 else 4
        )
        
        Y_prev = torch.zeros_like(Y)
        Y_prev[:, 1:] = Y[:, :-1]
        dA = torch.matmul(dX.unsqueeze(-1), Y_prev.conj().unsqueeze(-2))
        
        if ctx.is_diag:
            dA = torch.diagonal(dA, dim1=-2, dim2=-1)
            
        return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    