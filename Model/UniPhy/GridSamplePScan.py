import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_composition(grid_prev, img_prev, grid_curr, img_curr):
    """
    Compose (grid_prev, img_prev) with (grid_curr, img_curr).
    Ensures that out-of-bound coordinates remain out-of-bound.
    """
    # 1. 计算 grid_curr 的有效性 Mask
    # 如果 grid_curr 的任意一个分量超出 [-1, 1]，则认为该采样点无效
    # (N, H, W, 1)
    mask = (grid_curr.abs() <= 1).all(dim=-1, keepdim=True).to(grid_curr.dtype)

    # 2. Grid Composition: grid_new(x) = grid_prev(grid_curr(x))
    grid_prev_perm = grid_prev.permute(0, 3, 1, 2)
    
    # 使用 border padding 保证界内插值的连续性
    grid_composed = F.grid_sample(
        grid_prev_perm, 
        grid_curr, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )
    grid_composed = grid_composed.permute(0, 2, 3, 1)

    # 关键修复：利用 mask 强制处理出界点
    # 如果 grid_curr 出界，则 grid_composed 强制设为 -2.0 (一个足够远的值)
    # 这样在下一轮递归中，它依然是出界的
    out_of_bound_val = -2.0 * torch.ones_like(grid_composed)
    grid_composed = mask * grid_composed + (1 - mask) * out_of_bound_val

    # 3. Image Advection: img_new(x) = img_prev(grid_curr(x)) + img_curr(x)
    # Image 部分使用 zeros padding 即可，出界自动变黑
    img_warped = F.grid_sample(
        img_prev, 
        grid_curr, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    img_composed = img_warped + img_curr

    return grid_composed, img_composed


class GridSamplePScan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grids, images):
        """
        grids: (B, L, H, W, 2) values ideally in [-1, 1], but can handle outliers
        images: (B, L, C, H, W)
        """
        curr_grids = grids.clone()
        curr_images = images.clone()
        
        B, L, H, W, _ = grids.shape
        _, _, C, _, _ = images.shape

        step = 1
        while step < L:
            prev_grids_part = curr_grids[:, :-step].contiguous()
            prev_images_part = curr_images[:, :-step].contiguous()
            
            curr_grids_part = curr_grids[:, step:].contiguous()
            curr_images_part = curr_images[:, step:].contiguous()

            B_part, L_part = prev_grids_part.shape[:2]
            N_part = B_part * L_part
            
            flat_prev_grids = prev_grids_part.view(N_part, H, W, 2)
            flat_prev_images = prev_images_part.view(N_part, C, H, W)
            flat_curr_grids = curr_grids_part.view(N_part, H, W, 2)
            flat_curr_images = curr_images_part.view(N_part, C, H, W)

            next_grids_flat, next_images_flat = flow_composition(
                flat_prev_grids, 
                flat_prev_images, 
                flat_curr_grids, 
                flat_curr_images
            )

            next_grids_part = next_grids_flat.view(B_part, L_part, H, W, 2)
            next_images_part = next_images_flat.view(B_part, L_part, C, H, W)

            prefix_grids = curr_grids[:, :step]
            prefix_images = curr_images[:, :step]
            
            curr_grids = torch.cat([prefix_grids, next_grids_part], dim=1)
            curr_images = torch.cat([prefix_images, next_images_part], dim=1)

            step *= 2

        return curr_images


def pscan_flow(grids, images):
    scanner = GridSamplePScan()
    return scanner(grids, images)

