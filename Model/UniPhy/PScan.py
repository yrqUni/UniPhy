import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def combine_fn(ar, ai, xr, xi, br, bi, yr, yi, R: tl.constexpr):
    ar = tl.reshape(ar, (R, R))
    ai = tl.reshape(ai, (R, R))
    xr = tl.reshape(xr, (R, R))
    xi = tl.reshape(xi, (R, R))
    br = tl.reshape(br, (R, R))
    bi = tl.reshape(bi, (R, R))
    yr = tl.reshape(yr, (R, R))
    yi = tl.reshape(yi, (R, R))

    nar = tl.dot(br, ar) - tl.dot(bi, ai)
    nai = tl.dot(br, ai) + tl.dot(bi, ar)

    nxr = tl.dot(br, xr) - tl.dot(bi, xi) + yr
    nxi = tl.dot(br, xi) + tl.dot(bi, xr) + yi

    return tl.ravel(nar), tl.ravel(nai), tl.ravel(nxr), tl.ravel(nxi)

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

    ar = tl.load(a_base + idx_l[:, None] * stride_al + offs_mat[None, :], mask=mask_l[:, None], other=0.0)
    ai = tl.load(a_base + idx_l[:, None] * stride_al + offs_mat[None, :] + (R*R), mask=mask_l[:, None], other=0.0)

    xr_vec = tl.load(x_base + idx_l[:, None] * stride_xl + offs_vec[None, :], mask=mask_l[:, None], other=0.0)
    xi_vec = tl.load(x_base + idx_l[:, None] * stride_xl + offs_vec[None, :] + R, mask=mask_l[:, None], other=0.0)

    xr = tl.where(offs_mat[None, :] < R, xr_vec, 0.0)
    xi = tl.where(offs_mat[None, :] < R, xi_vec, 0.0)

    if REVERSE:
        eye = (offs_mat // R) == (offs_mat % R)
        ar = tl.where(offs_l[:, None] == 0, tl.where(eye, 1.0, 0.0), ar)
        ai = tl.where(offs_l[:, None] == 0, 0.0, ai)

    rar, rai, rxr, rxi = tl.associative_scan((ar, ai, xr, xi), axis=0, combine_fn=lambda ar, ai, xr, xi, br, bi, yr, yi: combine_fn(ar, ai, xr, xi, br, bi, yr, yi, R))

    y_base = Y_ptr + pid_b * stride_xb + pid_c * stride_xc
    res_xr = tl.reshape(rxr, (BLOCK_SIZE, R, R))[:, :, 0]
    res_xi = tl.reshape(rxi, (BLOCK_SIZE, R, R))[:, :, 0]
    tl.store(y_base + idx_l[:, None] * stride_xl + offs_vec[None, :], res_xr, mask=mask_l[:, None])
    tl.store(y_base + idx_l[:, None] * stride_xl + offs_vec[None, :] + R, res_xi, mask=mask_l[:, None])

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        ctx.is_diag = A.ndim == 4
        if ctx.is_diag: A = torch.diag_embed(A)
        B, L, C, R, _ = A.shape
        A, X, Y = A.contiguous(), X.contiguous(), torch.empty_like(X)
        pscan_kernel[(B, C)](torch.view_as_real(A), torch.view_as_real(X), torch.view_as_real(Y), A.stride(0), A.stride(1), A.stride(2), X.stride(0), X.stride(1), X.stride(2), L, R, triton.next_power_of_2(L), False, num_warps=8 if R >= 16 else 4)
        ctx.save_for_backward(A, X, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, X, Y = ctx.saved_tensors
        B, L, C, R, _ = A.shape
        AT = A.conj().transpose(-1, -2).contiguous()
        AP = torch.zeros_like(AT)
        AP[:, :-1], AP[:, -1] = AT[:, 1:], torch.eye(R, device=A.device, dtype=A.dtype).expand(B, C, R, R)
        grad_output, dX = grad_output.contiguous(), torch.empty_like(grad_output)
        pscan_kernel[(B, C)](torch.view_as_real(AP), torch.view_as_real(grad_output), torch.view_as_real(dX), AP.stride(0), AP.stride(1), AP.stride(2), grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), L, R, triton.next_power_of_2(L), True, num_warps=8 if R >= 16 else 4)
        YP = torch.zeros_like(Y)
        YP[:, 1:] = Y[:, :-1]
        dA = torch.matmul(dX.unsqueeze(-1), YP.conj().unsqueeze(-2))
        if ctx.is_diag: dA = torch.diagonal(dA, dim1=-2, dim2=-1)
        return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X): return _PScanFunction.apply(A, X)
    