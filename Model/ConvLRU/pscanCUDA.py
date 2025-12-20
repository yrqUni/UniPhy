import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void complex_mul(float ar, float ai, float br, float bi, float& out_r, float& out_i) {
    out_r = ar * br - ai * bi;
    out_i = ar * bi + ai * br;
}

__device__ __forceinline__ void combine(
    float ar_next, float ai_next, float xr_next, float xi_next,
    float ar_prev, float ai_prev, float xr_prev, float xi_prev,
    float& out_ar, float& out_ai, float& out_xr, float& out_xi
) {
    float prod_ar, prod_ai;
    complex_mul(ar_next, ai_next, ar_prev, ai_prev, prod_ar, prod_ai);
    float prod_xr, prod_xi;
    complex_mul(ar_next, ai_next, xr_prev, xi_prev, prod_xr, prod_xi);
    out_ar = prod_ar;
    out_ai = prod_ai;
    out_xr = prod_xr + xr_next;
    out_xi = prod_xi + xi_next;
}

__global__ void pscan_forward_kernel(
    const float* __restrict__ A_real, const float* __restrict__ A_imag,
    const float* __restrict__ X_real, const float* __restrict__ X_imag,
    float* __restrict__ Y_real, float* __restrict__ Y_imag,
    int L
) {
    extern __shared__ float shared_mem[];
    float* s_ar = shared_mem;
    float* s_ai = s_ar + 32;
    float* s_xr = s_ai + 32;
    float* s_xi = s_xr + 32;

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int seq_idx = blockIdx.x;
    int base_offset = seq_idx * L;

    float ar = (tid < L) ? A_real[base_offset + tid] : 1.0f;
    float ai = (tid < L) ? A_imag[base_offset + tid] : 0.0f;
    float xr = (tid < L) ? X_real[base_offset + tid] : 0.0f;
    float xi = (tid < L) ? X_imag[base_offset + tid] : 0.0f;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float shfl_ar = __shfl_up_sync(0xffffffff, ar, offset);
        float shfl_ai = __shfl_up_sync(0xffffffff, ai, offset);
        float shfl_xr = __shfl_up_sync(0xffffffff, xr, offset);
        float shfl_xi = __shfl_up_sync(0xffffffff, xi, offset);
        if (lane_id >= offset) {
            float nar, nai, nxr, nxi;
            combine(ar, ai, xr, xi, shfl_ar, shfl_ai, shfl_xr, shfl_xi, nar, nai, nxr, nxi);
            ar = nar; ai = nai; xr = nxr; xi = nxi;
        }
    }

    if (lane_id == 31) {
        s_ar[warp_id] = ar;
        s_ai[warp_id] = ai;
        s_xr[warp_id] = xr;
        s_xi[warp_id] = xi;
    }

    __syncthreads();

    if (warp_id == 0) {
        float w_ar = (tid < 32) ? s_ar[tid] : 1.0f;
        float w_ai = (tid < 32) ? s_ai[tid] : 0.0f;
        float w_xr = (tid < 32) ? s_xr[tid] : 0.0f;
        float w_xi = (tid < 32) ? s_xi[tid] : 0.0f;

        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float shfl_ar = __shfl_up_sync(0xffffffff, w_ar, offset);
            float shfl_ai = __shfl_up_sync(0xffffffff, w_ai, offset);
            float shfl_xr = __shfl_up_sync(0xffffffff, w_xr, offset);
            float shfl_xi = __shfl_up_sync(0xffffffff, w_xi, offset);
            if (lane_id >= offset) {
                float nar, nai, nxr, nxi;
                combine(w_ar, w_ai, w_xr, w_xi, shfl_ar, shfl_ai, shfl_xr, shfl_xi, nar, nai, nxr, nxi);
                w_ar = nar; w_ai = nai; w_xr = nxr; w_xi = nxi;
            }
        }
        if (tid < 32) {
            s_ar[tid] = w_ar;
            s_ai[tid] = w_ai;
            s_xr[tid] = w_xr;
            s_xi[tid] = w_xi;
        }
    }

    __syncthreads();

    if (warp_id > 0) {
        float base_ar = s_ar[warp_id - 1];
        float base_ai = s_ai[warp_id - 1];
        float base_xr = s_xr[warp_id - 1];
        float base_xi = s_xi[warp_id - 1];
        float nar, nai, nxr, nxi;
        combine(ar, ai, xr, xi, base_ar, base_ai, base_xr, base_xi, nar, nai, nxr, nxi);
        ar = nar; ai = nai; xr = nxr; xi = nxi;
    }

    if (tid < L) {
        Y_real[base_offset + tid] = xr;
        Y_imag[base_offset + tid] = xi;
    }
}

void pscan_forward_cuda(
    at::Tensor A_real, at::Tensor A_imag,
    at::Tensor X_real, at::Tensor X_imag,
    at::Tensor Y_real, at::Tensor Y_imag
) {
    int num_seqs = A_real.size(0);
    int L = A_real.size(1);
    
    int threads = 1024;
    int blocks = num_seqs;
    int shared_mem_size = 4 * 32 * sizeof(float);
    
    pscan_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        X_real.data_ptr<float>(), X_imag.data_ptr<float>(),
        Y_real.data_ptr<float>(), Y_imag.data_ptr<float>(),
        L
    );
}
"""

cpp_source = """
void pscan_forward_cuda(
    at::Tensor A_real, at::Tensor A_imag,
    at::Tensor X_real, at::Tensor X_imag,
    at::Tensor Y_real, at::Tensor Y_imag
);
"""

try:
    pscan_ext = load_inline(
        name='pscan_cuda_ext_v5',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['pscan_forward_cuda'],
        with_cuda=True,
        extra_cuda_cflags=['-O3', '--use_fast_math']
    )
except Exception as e:
    pscan_ext = None

class PScanCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X):
        input_shape = A.shape
        L = input_shape[1]
        
        if L > 1024:
            raise ValueError(f"CUDA implementation limited to L<=1024, got {L}")
        if pscan_ext is None:
            raise RuntimeError("CUDA extension not loaded.")

        A_in = A.transpose(1, -1).contiguous()
        X_in = X.transpose(1, -1).contiguous()
        
        A_flat = A_in.view(-1, L)
        X_flat = X_in.view(-1, L)
        
        A_real = A_flat.real.contiguous()
        A_imag = A_flat.imag.contiguous()
        X_real = X_flat.real.contiguous()
        X_imag = X_flat.imag.contiguous()
        
        Y_real = torch.empty_like(X_real)
        Y_imag = torch.empty_like(X_imag)
        
        pscan_ext.pscan_forward_cuda(A_real, A_imag, X_real, X_imag, Y_real, Y_imag)
        
        Y_flat = torch.complex(Y_real, Y_imag)
        Y = Y_flat.view(*A_in.shape)
        Y = Y.transpose(1, -1)
        
        ctx.save_for_backward(A, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        A, Y = ctx.saved_tensors
        A_conj = A.conj()
        
        grad_output_rev = grad_output.flip(1)
        A_conj_rev = A_conj.flip(1)
        
        slices_start = [slice(None)] * A.ndim
        slices_start[1] = slice(0, 1)
        slices_end = [slice(None)] * A.ndim
        slices_end[1] = slice(None, -1)
        
        A_rev_shifted = torch.cat([
            torch.zeros_like(A_conj_rev[tuple(slices_start)]), 
            A_conj_rev[tuple(slices_end)]
        ], dim=1)
        
        dX_rev = PScanCUDA.apply(A_rev_shifted, grad_output_rev)
        dX = dX_rev.flip(1)
        
        Y_prev = torch.cat([
            torch.zeros_like(Y[tuple(slices_start)]), 
            Y[tuple(slices_end)]
        ], dim=1)
        
        dA = dX * Y_prev.conj()
        return dA, dX

pscan = PScanCUDA.apply

def pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
    if not torch.cuda.is_available():
        return True
    device = 'cuda'
    dtype = torch.complex64
    
    print(f"Checking CUDA PScan with 5D Input (L={seq_length})...")
    
    A = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    X = torch.randn(batch_size, seq_length, channels, state_dim, device=device, dtype=dtype, requires_grad=True)
    
    try:
        Y_cuda = pscan(A, X)
    except Exception as e:
        print(f"CUDA Run Failed: {e}")
        return False
    
    Y_serial = torch.zeros_like(X)
    h = torch.zeros(batch_size, channels, state_dim, device=device, dtype=dtype)
    
    for t in range(seq_length):
        h = A[:, t] * h + X[:, t]
        Y_serial[:, t] = h
        
    max_diff = (Y_cuda - Y_serial).abs().max().item()
    if max_diff > 1e-4:
        print(f"Mismatch: {max_diff}")
        return False
        
    loss = Y_cuda.sum().abs()
    loss.backward()
    print("✅ PScan Check Passed.")
    return True
