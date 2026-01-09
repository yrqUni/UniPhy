import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float wrap_coord(float x) {
    float t = x + 1.0f;
    float r = fmodf(t, 2.0f);
    if (r < 0.0f) r += 2.0f;
    return r - 1.0f;
}

template <typename scalar_t>
__global__ void fused_pscan_kernel(
    const scalar_t* __restrict__ flows,
    const scalar_t* __restrict__ images,
    scalar_t* __restrict__ output,
    int B, int L, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = B * L * H * W;
    
    if (idx >= total_pixels) return;

    int tmp = idx;
    int w_idx = tmp % W; tmp /= W;
    int h_idx = tmp % H; tmp /= H;
    int t_idx = tmp % L; tmp /= L;
    int b_idx = tmp;

    int stride_flow_t = 2 * H * W;
    int stride_img_t  = C * H * W;
    int stride_plane  = H * W;

    const scalar_t* cur_flows = flows + b_idx * (L * stride_flow_t);
    const scalar_t* cur_imgs  = images + b_idx * (L * stride_img_t);
    scalar_t* cur_out   = output + b_idx * (L * stride_img_t);

    int flow_offset_base = t_idx * stride_flow_t + h_idx * W + w_idx;
    scalar_t flow_t_x = cur_flows[flow_offset_base];
    scalar_t flow_t_y = cur_flows[flow_offset_base + stride_plane];

    float base_x = (w_idx + 0.5f) * (2.0f / W) - 1.0f;
    float base_y = (h_idx + 0.5f) * (2.0f / H) - 1.0f;

    for (int c = 0; c < C; ++c) {
        scalar_t acc = 0;
        
        for (int k = 0; k <= t_idx; ++k) {
            int flow_k_offset = k * stride_flow_t + h_idx * W + w_idx;
            scalar_t flow_k_x = cur_flows[flow_k_offset];
            scalar_t flow_k_y = cur_flows[flow_k_offset + stride_plane];

            scalar_t rel_x = flow_t_x - flow_k_x;
            scalar_t rel_y = flow_t_y - flow_k_y;
            
            float gx = wrap_coord(base_x + rel_x);
            float gy = wrap_coord(base_y + rel_y);

            float ix = (gx + 1.0f) * 0.5f * W - 0.5f;
            float iy = (gy + 1.0f) * 0.5f * H - 0.5f;
            
            int x0 = static_cast<int>(floorf(ix));
            int y0 = static_cast<int>(floorf(iy));
            int x1 = x0 + 1; 
            int y1 = y0 + 1;
            
            float wx1 = ix - x0; 
            float wy1 = iy - y0;
            float wx0 = 1.0f - wx1; 
            float wy0 = 1.0f - wy1;
            
            const scalar_t* img_plane = cur_imgs + k * stride_img_t + c * stride_plane;
            
            scalar_t val = 0;
            if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) val += wx0 * wy0 * img_plane[y0 * W + x0];
            if (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W) val += wx1 * wy0 * img_plane[y0 * W + x1];
            if (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W) val += wx0 * wy1 * img_plane[y1 * W + x0];
            if (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W) val += wx1 * wy1 * img_plane[y1 * W + x1];
            
            acc += val;
        }
        
        int out_idx = t_idx * stride_img_t + c * stride_plane + h_idx * W + w_idx;
        cur_out[out_idx] = acc;
    }
}

torch::Tensor pscan_forward_cuda_impl(torch::Tensor flows, torch::Tensor images) {
    auto B = images.size(0);
    auto L = images.size(1);
    auto C = images.size(2);
    auto H = images.size(3);
    auto W = images.size(4);
    
    auto output = torch::zeros_like(images);
    
    int total_threads = B * L * H * W;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(flows.scalar_type(), "fused_pscan_kernel", ([&] {
        fused_pscan_kernel<scalar_t><<<blocks, threads>>>(
            flows.data_ptr<scalar_t>(),
            images.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, L, C, H, W
        );
    }));
    
    return output;
}
'''

cpp_source = r'''
torch::Tensor pscan_forward_cuda_impl(torch::Tensor flows, torch::Tensor images);
'''

fused_pscan = load_inline(
    name='fused_pscan_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['pscan_forward_cuda_impl'],
    verbose=False,
    extra_cuda_cflags=['-O3']
)

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
        
    def forward(self, flows, images):
        cum_flows = torch.cumsum(flows.float(), dim=1).to(flows.dtype)
        
        if not cum_flows.is_contiguous(): cum_flows = cum_flows.contiguous()
        if not images.is_contiguous(): images = images.contiguous()
        
        return fused_pscan.pscan_forward_cuda_impl(cum_flows, images)

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    
    return GridSamplePScan(mode=mode)(flows, images)

