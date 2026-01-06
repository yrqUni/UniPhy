import torch
import torch.nn as nn
import torch.nn.functional as F

def warp_flow(flow_prev, flow_curr):
    B, _, H, W = flow_prev.shape
    device = flow_prev.device
    dtype = flow_prev.dtype

    step_y = 2.0 / H
    step_x = 2.0 / W
    gy = torch.linspace(-1 + step_y/2, 1 - step_y/2, H, device=device, dtype=dtype)
    gx = torch.linspace(-1 + step_x/2, 1 - step_x/2, W, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    flow_curr_perm = flow_curr.permute(0, 2, 3, 1)
    sample_grid = base_grid + flow_curr_perm

    sample_grid_w = sample_grid[..., 0]
    sample_grid_w_wrapped = torch.remainder(sample_grid_w + 1, 2) - 1
    sample_grid_wrapped = torch.stack([sample_grid_w_wrapped, sample_grid[..., 1]], dim=-1)

    sampled_prev = F.grid_sample(
        flow_prev, 
        sample_grid_wrapped,
        mode='bilinear',
        padding_mode='border', 
        align_corners=False
    )

    flow_new = flow_curr + sampled_prev
    return flow_new

def warp_image(img_prev, flow_curr):
    B, _, H, W = img_prev.shape
    device = img_prev.device
    dtype = img_prev.dtype
    
    step_y = 2.0 / H
    step_x = 2.0 / W
    gy = torch.linspace(-1 + step_y/2, 1 - step_y/2, H, device=device, dtype=dtype)
    gx = torch.linspace(-1 + step_x/2, 1 - step_x/2, W, device=device, dtype=dtype)
        
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    flow_curr_perm = flow_curr.permute(0, 2, 3, 1)
    sample_grid = base_grid + flow_curr_perm
    
    sample_grid_w = sample_grid[..., 0]
    sample_grid_w_wrapped = torch.remainder(sample_grid_w + 1, 2) - 1
    sample_grid_wrapped = torch.stack([sample_grid_w_wrapped, sample_grid[..., 1]], dim=-1)
    
    img_warped = F.grid_sample(
        img_prev,
        sample_grid_wrapped,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    return img_warped

def flow_composition_residual(flow_prev, img_prev, flow_curr, img_curr):
    flow_combined = warp_flow(flow_prev, flow_curr)
    img_warped = warp_image(img_prev, flow_curr)
    img_combined = img_warped + img_curr
    return flow_combined, img_combined

class GridSamplePScan(nn.Module):
    def __init__(self):
        super().__init__()

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
                flat_curr_images
            )

            next_flows_part = next_flows_flat.view(B_part, L_part, 2, H, W)
            next_images_part = next_images_flat.view(B_part, L_part, C, H, W)

            prefix_flows = curr_flows[:, :step]
            prefix_images = curr_images[:, :step]
            
            curr_flows = torch.cat([prefix_flows, next_flows_part], dim=1)
            curr_images = torch.cat([prefix_images, next_images_part], dim=1)

            step *= 2

        return curr_images

def pscan_flow(flows, images):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan()
    return scanner(flows, images)

