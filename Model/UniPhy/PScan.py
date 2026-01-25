import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------
# Triton Kernels & Helpers
# --------------------------

@triton.jit
def complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@triton.jit
def complex_combine(ar, ai, xr, xi, br, bi, yr, yi):
    # Tuple: (A_prev, X_prev), (A_curr, X_curr) -> (ar, ai, xr, xi), (br, bi, yr, yi)
    # H_new = A_curr * H_prev + X_curr
    # A_new = A_curr * A_prev
    # X_new = A_curr * X_prev + X_curr
    new_ar, new_ai = complex_mul(br, bi, ar, ai)
    bx_r, bx_i = complex_mul(br, bi, xr, xi)
    new_xr = bx_r + yr
    new_xi = bx_i + yi
    return new_ar, new_ai, new_xr, new_xi

@triton.jit
def matrix_combine(a, x, b, y):
    # Tuple: (A_prev, X_prev), (A_curr, X_curr) -> (a, x), (b, y)
    # H_new = B @ H_prev + Y
    # A_new = B @ A
    # X_new = B @ X + Y
    new_a = tl.dot(b, a)
    bx = tl.dot(b, x)
    new_x = bx + y
    return new_a, new_x

def get_configs():
    return [
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ]

# --------------------------
# Complex Diagonal Kernel
# --------------------------

@triton.autotune(configs=get_configs(), key=["L"])
@triton.jit
def pscan_complex_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch, stride_time,
    L,
    BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    A_base = A_ptr + pid * stride_batch
    X_base = X_ptr + pid * stride_batch
    Y_base = Y_ptr + pid * stride_batch
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < L
    read_offs = tl.where(REVERSE, (L - 1 - offs), offs)
    
    # Load Complex A and X
    a_r = tl.load(A_base + read_offs * stride_time + 0, mask=mask, other=1.0)
    a_i = tl.load(A_base + read_offs * stride_time + 1, mask=mask, other=0.0)
    x_r = tl.load(X_base + read_offs * stride_time + 0, mask=mask, other=0.0)
    x_i = tl.load(X_base + read_offs * stride_time + 1, mask=mask, other=0.0)
    
    acc_ar, acc_ai, acc_xr, acc_xi = tl.associative_scan(
        (a_r, a_i, x_r, x_i), axis=0, combine_fn=complex_combine
    )
    
    tl.store(Y_base + read_offs * stride_time + 0, acc_xr, mask=mask)
    tl.store(Y_base + read_offs * stride_time + 1, acc_xi, mask=mask)

# --------------------------
# Matrix Kernel (Non-Diagonal)
# --------------------------

@triton.autotune(configs=get_configs(), key=["L", "DIM"])
@triton.jit
def pscan_matrix_kernel(
    A_ptr, X_ptr, Y_ptr,
    stride_batch_A, stride_time_A,
    stride_batch_X, stride_time_X,
    stride_batch_Y, stride_time_Y,
    L,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Pointers setup
    offs_t = tl.arange(0, BLOCK_SIZE)
    mask_t = offs_t < L
    read_offs_t = tl.where(REVERSE, (L - 1 - offs_t), offs_t)
    
    # Indices for matrix/vector loading
    # A shape: (BLOCK, DIM, DIM)
    # X shape: (BLOCK, DIM)
    
    r = tl.arange(0, DIM)
    
    # Pointers to A (Matrix): [Batch, L, DIM, DIM]
    # We load a block of matrices. 
    # Pointer math: base + t*stride_time + row*stride_row + col*stride_col
    # Assuming contiguous last dims: stride_row=DIM, stride_col=1
    ptr_A = A_ptr + pid * stride_batch_A + read_offs_t[:, None, None] * stride_time_A
    offs_A = r[None, :, None] * DIM + r[None, None, :]
    ptrs_A = ptr_A + offs_A
    
    # Pointers to X (Vector): [Batch, L, DIM]
    ptr_X = X_ptr + pid * stride_batch_X + read_offs_t[:, None] * stride_time_X
    offs_X = r[None, :]
    ptrs_X = ptr_X + offs_X

    # Load
    # Masking: we need to mask the time dimension. 
    # broadcast mask_t to (BLOCK, 1, 1) and (BLOCK, 1)
    mask_A = mask_t[:, None, None]
    mask_X = mask_t[:, None]
    
    # Identity matrix for padding A, Zeros for padding X
    eye = tl.zeros((DIM, DIM), dtype=tl.float32)
    for i in range(DIM):
        eye[i, i] = 1.0
        
    a_block = tl.load(ptrs_A, mask=mask_A, other=eye) # (BLOCK, DIM, DIM)
    x_block = tl.load(ptrs_X, mask=mask_X, other=0.0) # (BLOCK, DIM)
    
    # Scan
    acc_a, acc_x = tl.associative_scan(
        (a_block, x_block), axis=0, combine_fn=matrix_combine
    )
    
    # Store Y
    ptr_Y = Y_ptr + pid * stride_batch_Y + read_offs_t[:, None] * stride_time_Y
    ptrs_Y = ptr_Y + offs_X
    tl.store(ptrs_Y, acc_x, mask=mask_X)

# --------------------------
# PyTorch Function
# --------------------------

def next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

class _PScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        # Determine Mode
        # Matrix Mode: A is (..., D, D), X is (..., D)
        # Diagonal Mode: A is (..., 2) Complex, X is (..., 2) Complex
        
        is_matrix = (A.ndim >= 3) and (A.shape[-1] == A.shape[-2]) and (A.shape[-1] > 2 or A.shape[-1] != 2)
        # Note: If D=2, it could be ambiguous, but prompt implies 'chw' for matrix, so usually >2.
        # However, to be safe, if X has dim D and A has dim D,D, it's matrix.
        if A.shape[-1] == 2 and X.shape[-1] == 2 and A.ndim == X.ndim:
             is_matrix = False
        elif A.ndim == X.ndim + 1:
             is_matrix = True
        
        ctx.is_matrix = is_matrix
        ctx.shape_A_orig = A.shape
        ctx.shape_X_orig = X.shape

        if not is_matrix:
            # --- Diagonal Complex Mode ---
            if A.ndim == X.ndim - 1:
                A = A.unsqueeze(-1)
            if A.shape != X.shape:
                A, X = torch.broadcast_tensors(A, X)
            A, X = A.contiguous(), X.contiguous()
            
            L = X.shape[1]
            # Flatten: (Batch * Channel, L, 2)
            A_in = A.transpose(1, -1).contiguous() # (..., 2, L) -> (..., L, 2) actually user code did transposes differently
            # User code assumed last dim is Real/Imag. 
            # Let's standardize input to (B_flat, L, 2)
            # A shape is [B, L, C, 2].
            
            # Reshape to (Batch*Channel, L, 2)
            # We assume dim 1 is L.
            batch_dims = A.shape[0] * A.shape[2] 
            A_flat = A.view(batch_dims, L, 2)
            X_flat = X.view(batch_dims, L, 2)

            Y_flat = torch.empty_like(X_flat)
            BLOCK_SIZE = max(32, next_power_of_2(L))
            
            pscan_complex_kernel[(batch_dims,)](
                A_flat, X_flat, Y_flat,
                A_flat.stride(0), A_flat.stride(1),
                L, BLOCK_SIZE, False
            )
            
            Y = Y_flat.view(*A.shape)
            ctx.save_for_backward(A, Y)
            return Y

        else:
            # --- Matrix Real Mode ---
            # A: [B, L, C, D, D] -> Flatten to [B*C, L, D, D]
            # X: [B, L, C, D]    -> Flatten to [B*C, L, D]
            
            # Ensure Contiguous
            A = A.contiguous()
            X = X.contiguous()
            
            B, L, C, D, _ = A.shape
            batch_dim = B * C
            
            A_flat = A.view(batch_dim, L, D, D)
            X_flat = X.view(batch_dim, L, D)
            Y_flat = torch.empty_like(X_flat)
            
            BLOCK_SIZE = max(32, next_power_of_2(L))
            
            pscan_matrix_kernel[(batch_dim,)](
                A_flat, X_flat, Y_flat,
                A_flat.stride(0), A_flat.stride(1),
                X_flat.stride(0), X_flat.stride(1),
                Y_flat.stride(0), Y_flat.stride(1),
                L, D, BLOCK_SIZE, False
            )
            
            Y = Y_flat.view(*X.shape)
            ctx.save_for_backward(A, Y)
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, Y = ctx.saved_tensors
        is_matrix = ctx.is_matrix
        
        if not is_matrix:
            # --- Diagonal Complex Backward ---
            # A: (B, L, C, 2), Y: (B, L, C, 2)
            # A_conj: Complex Conjugate of A
            A_conj = A.clone()
            A_conj[..., 1] = -A_conj[..., 1]
            
            # Shift A for reverse scan: A_bar_t = A_{t+1}^*
            # Pad last time step with 0
            A_prep = torch.cat([A_conj[:, 1:], torch.zeros_like(A_conj[:, 0:1])], dim=1)
            
            grad_output = grad_output.contiguous()
            L = A.shape[1]
            batch_dims = A.shape[0] * A.shape[2]
            
            A_flat = A_prep.view(batch_dims, L, 2)
            X_flat = grad_output.view(batch_dims, L, 2)
            dX_flat = torch.empty_like(X_flat)
            
            BLOCK_SIZE = max(32, next_power_of_2(L))
            
            # Reverse Scan to get dX (dL/dh)
            # Logic: dX_t = dL/dh_t_output + A_{t+1}^* * dX_{t+1}
            # This is exactly pscan(A_prep, grad_output) in reverse
            pscan_complex_kernel[(batch_dims,)](
                A_flat, X_flat, dX_flat,
                A_flat.stride(0), A_flat.stride(1),
                L, BLOCK_SIZE, True
            )
            
            dX = dX_flat.view(*grad_output.shape)
            
            # Compute dA
            # dA_t = dX_t * Y_{t-1}^*
            # Shift Y
            Y_prev = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
            Y_prev_conj = Y_prev.clone()
            Y_prev_conj[..., 1] = -Y_prev_conj[..., 1]
            
            # Complex multiplication elementwise
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            dx_r, dx_i = dX[..., 0], dX[..., 1]
            yp_r, yp_i = Y_prev_conj[..., 0], Y_prev_conj[..., 1]
            
            da_r = dx_r * yp_r - dx_i * yp_i
            da_i = dx_r * yp_i + dx_i * yp_r
            dA = torch.stack([da_r, da_i], dim=-1)
            
            # Handle broadcasting reduction if necessary
            if dA.shape != ctx.shape_A_orig:
                for i, dim in enumerate(ctx.shape_A_orig):
                    if dim == 1: dA = dA.sum(dim=i, keepdim=True)
            if dX.shape != ctx.shape_X_orig:
                for i, dim in enumerate(ctx.shape_X_orig):
                    if dim == 1: dX = dX.sum(dim=i, keepdim=True)
                    
            return dA, dX
            
        else:
            # --- Matrix Real Backward ---
            # A: (B, L, C, D, D), Y: (B, L, C, D)
            # grad_output (dX_next): (B, L, C, D)
            
            # Reverse Scan inputs:
            # State: dX (gradient of hidden state)
            # Transition: A^T (Transpose of A)
            # Input: grad_output (gradient of current step output)
            
            # A_rev = A_{t+1}^T. 
            # Shift time:
            A_shift = torch.cat([A[:, 1:], torch.zeros_like(A[:, 0:1])], dim=1)
            A_prep = A_shift.transpose(-1, -2).contiguous() # Transpose last two dims
            
            L = A.shape[1]
            B, _, C, D, _ = A.shape
            batch_dim = B * C
            
            A_flat = A_prep.view(batch_dim, L, D, D)
            G_flat = grad_output.contiguous().view(batch_dim, L, D)
            dX_flat = torch.empty_like(G_flat)
            
            BLOCK_SIZE = max(32, next_power_of_2(L))
            
            pscan_matrix_kernel[(batch_dim,)](
                A_flat, G_flat, dX_flat,
                A_flat.stride(0), A_flat.stride(1),
                G_flat.stride(0), G_flat.stride(1),
                dX_flat.stride(0), dX_flat.stride(1),
                L, D, BLOCK_SIZE, True # REVERSE=True
            )
            
            dX = dX_flat.view(*grad_output.shape)
            
            # Compute dA
            # dA_t = dX_t (outer product) Y_{t-1}^T
            # dX_t is (B, L, C, D, 1) effectively
            # Y_{t-1} is (B, L, C, 1, D) effectively
            
            Y_prev = torch.cat([torch.zeros_like(Y[:, 0:1]), Y[:, :-1]], dim=1)
            
            # dA = dX.unsqueeze(-1) @ Y_prev.unsqueeze(-2)
            dA = torch.matmul(dX.unsqueeze(-1), Y_prev.unsqueeze(-2))
            
            # Handle broadcasting reduction
            if dA.shape != ctx.shape_A_orig:
                dims_to_sum = []
                for i, (orig, curr) in enumerate(zip(ctx.shape_A_orig, dA.shape)):
                     if orig == 1 and curr > 1:
                         dims_to_sum.append(i)
                if dims_to_sum:
                    dA = dA.sum(dim=dims_to_sum, keepdim=True)
            
            if dX.shape != ctx.shape_X_orig:
                dims_to_sum_x = []
                for i, (orig, curr) in enumerate(zip(ctx.shape_X_orig, dX.shape)):
                     if orig == 1 and curr > 1:
                         dims_to_sum_x.append(i)
                if dims_to_sum_x:
                    dX = dX.sum(dim=dims_to_sum_x, keepdim=True)
                    
            return dA, dX

class PScanTriton(nn.Module):
    def forward(self, A, X):
        return _PScanFunction.apply(A, X)
    