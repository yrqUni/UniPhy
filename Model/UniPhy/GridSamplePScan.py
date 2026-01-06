import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _warp_add_fwd_kernel(
    x_ptr, flow_ptr, res_ptr, out_ptr,
    B, C, H, W,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_f_b, stride_f_c, stride_f_h, stride_f_w,
    stride_r_b, stride_r_c, stride_r_h, stride_r_w,
    stride_o_b, stride_o_c, stride_o_h, stride_o_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (B * H * W)

    idx_w = offs % W
    tmp = offs // W
    idx_h = tmp % H
    idx_b = tmp // H

    step_h = 2.0 / H
    step_w = 2.0 / W
    
    base_y = idx_h * step_h - 1.0 + (step_h * 0.5)
    base_x = idx_w * step_w - 1.0 + (step_w * 0.5)

    off_f_base = idx_b * stride_f_b + idx_h * stride_f_h + idx_w * stride_f_w
    flow_x = tl.load(flow_ptr + off_f_base + 0 * stride_f_c, mask=mask, other=0.0)
    flow_y = tl.load(flow_ptr + off_f_base + 1 * stride_f_c, mask=mask, other=0.0)

    grid_x = base_x + flow_x
    grid_y = base_y + flow_y

    grid_x = (grid_x + 1.0) % 2.0 - 1.0

    real_x = (grid_x + 1.0) * (W * 0.5) - 0.5
    real_y = (grid_y + 1.0) * (H * 0.5) - 0.5

    x0 = tl.math.floor(real_x)
    y0 = tl.math.floor(real_y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = real_x - x0
    wx0 = 1.0 - wx1
    wy1 = real_y - y0
    wy0 = 1.0 - wy1

    w_tl = wx0 * wy0
    w_tr = wx1 * wy0
    w_bl = wx0 * wy1
    w_br = wx1 * wy1

    ix0 = x0.to(tl.int32)
    iy0 = y0.to(tl.int32)
    ix1 = x1.to(tl.int32)
    iy1 = y1.to(tl.int32)

    valid_tl = (ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H)
    valid_tr = (ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H)
    valid_bl = (ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H)
    valid_br = (ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H)

    off_in_base = idx_b * stride_x_b
    off_res_base = idx_b * stride_r_b + idx_h * stride_r_h + idx_w * stride_r_w
    off_out_base = idx_b * stride_o_b + idx_h * stride_o_h + idx_w * stride_o_w

    for c in range(C):
        off_c_x = c * stride_x_c
        
        v_tl = tl.load(x_ptr + off_in_base + off_c_x + iy0 * stride_x_h + ix0 * stride_x_w, mask=mask & valid_tl, other=0.0)
        v_tr = tl.load(x_ptr + off_in_base + off_c_x + iy0 * stride_x_h + ix1 * stride_x_w, mask=mask & valid_tr, other=0.0)
        v_bl = tl.load(x_ptr + off_in_base + off_c_x + iy1 * stride_x_h + ix0 * stride_x_w, mask=mask & valid_bl, other=0.0)
        v_br = tl.load(x_ptr + off_in_base + off_c_x + iy1 * stride_x_h + ix1 * stride_x_w, mask=mask & valid_br, other=0.0)

        interpolated = w_tl * v_tl + w_tr * v_tr + w_bl * v_bl + w_br * v_br

        res = tl.load(res_ptr + off_res_base + c * stride_r_c, mask=mask, other=0.0)
        
        tl.store(out_ptr + off_out_base + c * stride_o_c, interpolated + res, mask=mask)

class TritonWarpAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, flow, residual):
        B, C, H, W = input.shape
        ctx.save_for_backward(input, flow)
        
        output = torch.empty_like(input)
        
        grid = lambda META: (triton.cdiv(B * H * W, META['BLOCK_SIZE']), )
        
        _warp_add_fwd_kernel[grid](
            input, flow, residual, output,
            B, C, H, W,
            *input.stride(),
            *flow.stride(),
            *residual.stride(),
            *output.stride(),
            BLOCK_SIZE=512
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, flow = ctx.saved_tensors
        
        with torch.enable_grad():
            input_tmp = input.detach().requires_grad_(True)
            flow_tmp = flow.detach().requires_grad_(True)
            
            B, _, H, W = input.shape
            gy = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=input.device, dtype=input.dtype)
            gx = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=input.device, dtype=input.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
            base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            flow_perm = flow_tmp.permute(0, 2, 3, 1)
            grid = base_grid + flow_perm
            
            grid_w = grid[..., 0]
            grid_w = torch.remainder(grid_w + 1, 2) - 1
            grid = torch.stack([grid_w, grid[..., 1]], dim=-1)
            
            out = F.grid_sample(input_tmp, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            
            out.backward(grad_output)
            
            grad_input = input_tmp.grad
            grad_flow = flow_tmp.grad
            grad_residual = grad_output

        return grad_input, grad_flow, grad_residual

def _warp_torch(input, flow, residual, mode):
    B, _, H, W = input.shape
    gy = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=input.device, dtype=input.dtype)
    gx = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=input.device, dtype=input.dtype)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    flow_perm = flow.permute(0, 2, 3, 1)
    grid = base_grid + flow_perm
    
    grid_w = grid[..., 0]
    grid_w = torch.remainder(grid_w + 1, 2) - 1
    grid = torch.stack([grid_w, grid[..., 1]], dim=-1)
    
    out = F.grid_sample(input, grid, mode=mode, padding_mode='zeros', align_corners=False)
    
    if residual is not None:
        out = out + residual
    return out

def warp_flow(flow_prev, flow_curr, mode='bilinear'):
    if mode == 'bilinear':
        return TritonWarpAdd.apply(flow_prev, flow_curr, flow_curr)
    else:
        return _warp_torch(flow_prev, flow_curr, flow_curr, mode)

def warp_image(img_prev, flow_curr, mode='bilinear'):
    if mode == 'bilinear':
        zero_res = torch.zeros_like(img_prev)
        return TritonWarpAdd.apply(img_prev, flow_curr, zero_res)
    else:
        return _warp_torch(img_prev, flow_curr, None, mode)

def flow_composition_residual(flow_prev, img_prev, flow_curr, img_curr, mode='bilinear'):
    if mode == 'bilinear':
        flow_combined = TritonWarpAdd.apply(flow_prev, flow_curr, flow_curr)
        img_combined  = TritonWarpAdd.apply(img_prev,  flow_curr, img_curr)
    else:
        flow_combined = _warp_torch(flow_prev, flow_curr, flow_curr, mode)
        img_combined  = _warp_torch(img_prev,  flow_curr, img_curr, mode)
        
    return flow_combined, img_combined

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, flows, images):
        curr_flows = flows.clone()
        curr_images = images.clone()
        
        B, L, _, H, W = flows.shape
        _, _, C, _, _ = images.shape

        step = 1
        while step < L:
            prev_flows_part = curr_flows[:, :-step].contiguous()
            prev_images_part = curr_images[:, :-step].contiguous()
            
            curr_flows_part = curr_flows[:, step:].contiguous()
            curr_images_part = curr_images[:, step:].contiguous()

            B_part, L_part = prev_flows_part.shape[:2]
            N_part = B_part * L_part
            
            flat_prev_flows = prev_flows_part.view(N_part, 2, H, W)
            flat_prev_images = prev_images_part.view(N_part, C, H, W)
            flat_curr_flows = curr_flows_part.view(N_part, 2, H, W)
            flat_curr_images = curr_images_part.view(N_part, C, H, W)

            next_flows_flat, next_images_flat = flow_composition_residual(
                flat_prev_flows, 
                flat_prev_images, 
                flat_curr_flows, 
                flat_curr_images,
                mode=self.mode
            )

            next_flows_part = next_flows_flat.view(B_part, L_part, 2, H, W)
            next_images_part = next_images_flat.view(B_part, L_part, C, H, W)

            prefix_flows = curr_flows[:, :step]
            prefix_images = curr_images[:, :step]
            
            curr_flows = torch.cat([prefix_flows, next_flows_part], dim=1)
            curr_images = torch.cat([prefix_images, next_images_part], dim=1)

            step *= 2

        return curr_images

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan(mode=mode)
    return scanner(flows, images)

