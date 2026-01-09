import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pscan_bilinear_kernel(
    flows_ptr,
    images_ptr,
    output_ptr,
    B, L, C, H, W,
    stride_flow_b, stride_flow_l, stride_flow_d, stride_flow_h, stride_flow_w,
    stride_img_b, stride_img_l, stride_img_c, stride_img_h, stride_img_w,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    total_pixels = H * W
    num_blocks = (total_pixels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    bid = pid // (L * num_blocks)
    rem = pid % (L * num_blocks)
    tid = rem // num_blocks
    block_idx = rem % num_blocks
    
    offsets = block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < total_pixels

    w_idx = offsets % W
    h_idx = offsets // W

    base_x = (w_idx + 0.5) * (2.0 / W) - 1.0
    base_y = (h_idx + 0.5) * (2.0 / H) - 1.0

    flow_t_ptr_x = flows_ptr + bid * stride_flow_b + tid * stride_flow_l + 0 * stride_flow_d + h_idx * stride_flow_h + w_idx * stride_flow_w
    flow_t_ptr_y = flows_ptr + bid * stride_flow_b + tid * stride_flow_l + 1 * stride_flow_d + h_idx * stride_flow_h + w_idx * stride_flow_w

    flow_t_x = tl.load(flow_t_ptr_x, mask=mask, other=0.0)
    flow_t_y = tl.load(flow_t_ptr_y, mask=mask, other=0.0)

    for c in range(C):
        acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
        
        for k in range(tid + 1):
            flow_k_ptr_x = flows_ptr + bid * stride_flow_b + k * stride_flow_l + 0 * stride_flow_d + h_idx * stride_flow_h + w_idx * stride_flow_w
            flow_k_ptr_y = flows_ptr + bid * stride_flow_b + k * stride_flow_l + 1 * stride_flow_d + h_idx * stride_flow_h + w_idx * stride_flow_w
            
            flow_k_x = tl.load(flow_k_ptr_x, mask=mask, other=0.0)
            flow_k_y = tl.load(flow_k_ptr_y, mask=mask, other=0.0)

            rel_x = flow_t_x - flow_k_x
            rel_y = flow_t_y - flow_k_y

            grid_x = base_x + rel_x
            grid_y = base_y + rel_y
            
            grid_x = ((grid_x + 1.0) % 2.0) - 1.0
            grid_x = tl.where(grid_x < -1.0, grid_x + 2.0, grid_x)
            
            grid_y = ((grid_y + 1.0) % 2.0) - 1.0
            grid_y = tl.where(grid_y < -1.0, grid_y + 2.0, grid_y)

            real_x = (grid_x + 1.0) * 0.5 * W - 0.5
            real_y = (grid_y + 1.0) * 0.5 * H - 0.5

            x0 = tl.math.floor(real_x).to(tl.int32)
            y0 = tl.math.floor(real_y).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            wa = (x1 - real_x) * (y1 - real_y)
            wb = (x1 - real_x) * (real_y - y0)
            wc = (real_x - x0) * (y1 - real_y)
            wd = (real_x - x0) * (real_y - y0)

            img_base = images_ptr + bid * stride_img_b + k * stride_img_l + c * stride_img_c

            val_a = tl.load(img_base + y0 * stride_img_h + x0 * stride_img_w, mask=mask & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H), other=0.0)
            val_b = tl.load(img_base + y1 * stride_img_h + x0 * stride_img_w, mask=mask & (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H), other=0.0)
            val_c = tl.load(img_base + y0 * stride_img_h + x1 * stride_img_w, mask=mask & (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H), other=0.0)
            val_d = tl.load(img_base + y1 * stride_img_h + x1 * stride_img_w, mask=mask & (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H), other=0.0)

            acc += wa * val_a + wb * val_b + wc * val_c + wd * val_d

        out_ptr = output_ptr + bid * stride_img_b + tid * stride_img_l + c * stride_img_c + h_idx * stride_img_h + w_idx * stride_img_w
        tl.store(out_ptr, acc, mask=mask)

class PScanTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flows, images):
        cum_flows = torch.cumsum(flows.float(), dim=1).to(flows.dtype)
        
        B, L, C, H, W = images.shape
        output = torch.empty_like(images)
        
        BLOCK_SIZE_N = 256
        grid = (B * L * ((H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),)

        fused_pscan_bilinear_kernel[grid](
            cum_flows, images, output,
            B, L, C, H, W,
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        
        ctx.save_for_backward(flows, images)
        ctx.cum_flows = cum_flows
        return output

    @staticmethod
    def backward(ctx, grad_output):
        flows, images = ctx.saved_tensors
        cum_flows = ctx.cum_flows
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype

        grad_images = torch.zeros_like(images)
        grad_cum_flows_acc = torch.zeros_like(cum_flows)
        
        step_y, step_x = 2.0 / H, 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        base_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, H, W, 2)

        for delta in range(L):
            valid_len = L - delta
            
            flow_target = cum_flows[:, delta:]
            flow_source = cum_flows[:, :valid_len]
            rel_flow = flow_target - flow_source
            
            grid = base_grid + rel_flow.permute(0, 1, 3, 4, 2)
            grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0
            
            img_slice = images[:, :valid_len].detach().requires_grad_(True)
            grid_slice = grid.detach().requires_grad_(True)
            
            warped = nn.functional.grid_sample(
                img_slice.reshape(-1, C, H, W),
                grid_slice.reshape(-1, H, W, 2),
                mode='bilinear', padding_mode='zeros', align_corners=False
            ).view(B, valid_len, C, H, W)
            
            grad_out_slice = grad_output[:, delta:]
            torch.autograd.backward([warped], [grad_out_slice])
            
            grad_images[:, :valid_len] += img_slice.grad
            
            d_rel_flow = grid_slice.grad.view(B, valid_len, H, W, 2).permute(0, 1, 4, 2, 3)
            grad_cum_flows_acc[:, delta:] += d_rel_flow
            grad_cum_flows_acc[:, :valid_len] -= d_rel_flow

        grad_flows = torch.flip(torch.cumsum(torch.flip(grad_cum_flows_acc, [1]), dim=1), [1])
        
        return grad_flows, grad_images

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        assert mode == 'bilinear', "Triton kernel currently implements bilinear only"

    def forward(self, flows, images):
        return PScanTritonFunction.apply(flows, images)

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    return GridSamplePScan(mode=mode)(flows, images)

