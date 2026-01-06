import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 使用 JIT 编译融合坐标计算、流场相加和周期性包裹逻辑
# 这比纯 Python 快，且比 Triton Kernel 更稳定
@torch.jit.script
def gen_grid_fused(B: int, H: int, W: int, flow: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 1. Generate Base Grid
    # grid_y: [H, 1]
    # grid_x: [1, W]
    step_y = 2.0 / float(H)
    step_x = 2.0 / float(W)
    
    # linspace(-1 + step/2, 1 - step/2, steps)
    start_y = -1.0 + step_y * 0.5
    start_x = -1.0 + step_x * 0.5
    
    grid_y = torch.arange(H, device=device, dtype=dtype) * step_y + start_y
    grid_x = torch.arange(W, device=device, dtype=dtype) * step_x + start_x
    
    # 2. Add Flow (Broadcasting)
    # flow: [B, 2, H, W] -> permute to [B, H, W, 2]
    # x = base_x + flow_x
    # y = base_y + flow_y
    flow_perm = flow.permute(0, 2, 3, 1)
    
    # In-place optimizations usually risky with autograd, using standard ops
    # Expansion: [1, W] + [B, H, W] -> [B, H, W]
    final_x = grid_x.view(1, 1, -1) + flow_perm[..., 0]
    final_y = grid_y.view(1, -1, 1) + flow_perm[..., 1]
    
    # 3. Periodic Wrap for W (Longitude)
    # (x + 1) % 2 - 1
    final_x = torch.remainder(final_x + 1.0, 2.0) - 1.0
    
    return torch.stack([final_x, final_y], dim=-1)

def warp_flow(flow_prev: torch.Tensor, flow_curr: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    B, _, H, W = flow_prev.shape
    # JIT Fused Grid Gen
    grid = gen_grid_fused(B, H, W, flow_curr, flow_curr.device, flow_curr.dtype)
    
    # Native Optimized Sampling
    # padding_mode='zeros' is mathematically required for parallel scan associativity
    sampled_prev = F.grid_sample(
        flow_prev, 
        grid,
        mode=mode,
        padding_mode='zeros', 
        align_corners=False
    )
    return flow_curr + sampled_prev

def warp_image(img_prev: torch.Tensor, flow_curr: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    B, _, H, W = img_prev.shape
    grid = gen_grid_fused(B, H, W, flow_curr, flow_curr.device, flow_curr.dtype)
    
    return F.grid_sample(
        img_prev,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=False
    )

def flow_composition_residual(flow_prev: torch.Tensor, img_prev: torch.Tensor, flow_curr: torch.Tensor, img_curr: torch.Tensor, mode: str = 'bilinear') -> Tuple[torch.Tensor, torch.Tensor]:
    # Shared Grid Generation for both image and flow warping to save 50% grid compute
    B, _, H, W = flow_prev.shape
    grid = gen_grid_fused(B, H, W, flow_curr, flow_curr.device, flow_curr.dtype)
    
    flow_sampled = F.grid_sample(flow_prev, grid, mode=mode, padding_mode='zeros', align_corners=False)
    img_sampled = F.grid_sample(img_prev, grid, mode=mode, padding_mode='zeros', align_corners=False)
    
    return flow_curr + flow_sampled, img_curr + img_sampled

class GridSamplePScan(nn.Module):
    def __init__(self, mode: str = 'bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, flows: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # flows: [B, L, 2, H, W]
        # images: [B, L, C, H, W]
        curr_flows = flows
        curr_images = images
        
        B, L, _, H, W = flows.shape
        _, _, C, _, _ = images.shape

        # Parallel Scan (Log-Step)
        step = 1
        while step < L:
            # Slicing logic
            prev_flows_part = curr_flows[:, :-step].contiguous()
            prev_images_part = curr_images[:, :-step].contiguous()
            
            curr_flows_part = curr_flows[:, step:].contiguous()
            curr_images_part = curr_images[:, step:].contiguous()

            # Batch Processing: Flatten B and L dimensions
            # This is crucial for saturating the GPU
            B_part, L_part = prev_flows_part.shape[:2]
            N_part = B_part * L_part
            
            flat_prev_flows = prev_flows_part.view(N_part, 2, H, W)
            flat_prev_images = prev_images_part.view(N_part, C, H, W)
            flat_curr_flows = curr_flows_part.view(N_part, 2, H, W)
            flat_curr_images = curr_images_part.view(N_part, C, H, W)

            # Fused Composition Step
            next_flows_flat, next_images_flat = flow_composition_residual(
                flat_prev_flows, 
                flat_prev_images, 
                flat_curr_flows, 
                flat_curr_images,
                mode=self.mode
            )

            # Reshape back to [B, L, ...]
            next_flows_part = next_flows_flat.view(B_part, L_part, 2, H, W)
            next_images_part = next_images_flat.view(B_part, L_part, C, H, W)

            # Concatenate: Original prefix + New suffix
            # Note: We allocate new tensors here. In-place modification is tricky with autograd.
            prefix_flows = curr_flows[:, :step]
            prefix_images = curr_images[:, :step]
            
            curr_flows = torch.cat([prefix_flows, next_flows_part], dim=1)
            curr_images = torch.cat([prefix_images, next_images_part], dim=1)

            step *= 2

        return curr_images

def pscan_flow(flows: torch.Tensor, images: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    # Ensure channel dim is at 2 for flows [B, L, 2, H, W]
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan(mode=mode)
    return scanner(flows, images)

