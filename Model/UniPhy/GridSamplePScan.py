import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache

@lru_cache(maxsize=16)
def get_base_grid(B, H, W, device, dtype):
    # 改进: 明确指定 dtype，防止 AMP 环境下精度不匹配
    step_y = 2.0 / H
    step_x = 2.0 / W
    
    start_y = -1.0 + step_y * 0.5
    start_x = -1.0 + step_x * 0.5
    
    grid_y = torch.linspace(start_y, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
    grid_x = torch.linspace(start_x, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
    
    grid_y = grid_y.view(1, H, 1)
    grid_x = grid_x.view(1, 1, W)
    
    return grid_y, grid_x

def warp_common(flow, B, H, W, wrap_x=True):
    # 改进: 增加 wrap_x 参数，控制是否进行经度循环
    base_grid_y, base_grid_x = get_base_grid(B, H, W, flow.device, flow.dtype)
    
    flow_perm = flow.permute(0, 2, 3, 1)
    
    final_x = base_grid_x + flow_perm[..., 0]
    final_y = base_grid_y + flow_perm[..., 1]
    
    if wrap_x:
        # 仅在需要时执行循环边界处理
        final_x = torch.remainder(final_x + 1.0, 2.0) - 1.0
    
    return torch.stack([final_x, final_y], dim=-1)

def warp_flow(flow_prev, flow_curr, mode='bilinear', padding_mode='zeros', wrap_x=True):
    B, _, H, W = flow_prev.shape
    grid = warp_common(flow_curr, B, H, W, wrap_x=wrap_x)
    
    sampled_prev = F.grid_sample(
        flow_prev, 
        grid,
        mode=mode,
        padding_mode=padding_mode, 
        align_corners=False
    )
    return flow_curr + sampled_prev

def warp_image(img_prev, flow_curr, mode='bilinear', padding_mode='zeros', wrap_x=True):
    B, _, H, W = img_prev.shape
    grid = warp_common(flow_curr, B, H, W, wrap_x=wrap_x)
    
    return F.grid_sample(
        img_prev,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    )

def flow_composition_residual(flow_prev, img_prev, flow_curr, img_curr, mode='bilinear', padding_mode='zeros', wrap_x=True):
    # 改进: 支持传递 padding_mode 和 wrap_x
    B, _, H, W = flow_prev.shape
    grid = warp_common(flow_curr, B, H, W, wrap_x=wrap_x)
    
    flow_sampled = F.grid_sample(flow_prev, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    img_sampled = F.grid_sample(img_prev, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    return flow_curr + flow_sampled, img_curr + img_sampled

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear', padding_mode='zeros', wrap_x=True):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.wrap_x = wrap_x

    def forward(self, flows, images):
        # 改进: 显式 clone 一次，后续全部原地操作 (In-place)
        curr_flows = flows.clone()
        curr_images = images.clone()
        
        B, L, _, H, W = flows.shape
        _, _, C, _, _ = images.shape

        step = 1
        while step < L:
            # 使用切片引用，避免 .contiguous() 带来的不必要拷贝（除非 stride 不连续）
            # reshape 通常能处理大部分情况，比 view 更灵活
            
            # Read-only part (Previous State)
            prev_flows_part = curr_flows[:, :-step]
            prev_images_part = curr_images[:, :-step]
            
            # Write target (Current State that needs update)
            curr_flows_part = curr_flows[:, step:]
            curr_images_part = curr_images[:, step:]

            B_part, L_part = prev_flows_part.shape[:2]
            N_part = B_part * L_part
            
            # 变形为 (N, C, H, W) 用于 grid_sample
            flat_prev_flows = prev_flows_part.reshape(N_part, 2, H, W)
            flat_prev_images = prev_images_part.reshape(N_part, C, H, W)
            flat_curr_flows = curr_flows_part.reshape(N_part, 2, H, W)
            flat_curr_images = curr_images_part.reshape(N_part, C, H, W)

            next_flows_flat, next_images_flat = flow_composition_residual(
                flat_prev_flows, 
                flat_prev_images, 
                flat_curr_flows, 
                flat_curr_images,
                mode=self.mode,
                padding_mode=self.padding_mode,
                wrap_x=self.wrap_x
            )

            # 改进: 原地更新 (In-place update)，完全移除了 torch.cat
            # 这不会破坏 Autograd，因为 next_... 是新计算的张量，且写入位置与读取位置在计算图逻辑上是安全的
            curr_flows[:, step:] = next_flows_flat.view(B_part, L_part, 2, H, W)
            curr_images[:, step:] = next_images_flat.view(B_part, L_part, C, H, W)

            step *= 2

        return curr_images

def pscan_flow(flows, images, mode='bilinear', padding_mode='zeros', wrap_x=True):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    # 传递新增的配置参数
    scanner = GridSamplePScan(mode=mode, padding_mode=padding_mode, wrap_x=wrap_x)
    return scanner(flows, images)

