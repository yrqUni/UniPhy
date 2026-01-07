import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        if mode not in ['bilinear', 'nearest']:
            raise ValueError(f"mode must be 'bilinear' or 'nearest', got {mode}")
        self.mode = mode

    def get_base_grid(self, B, H, W, device, dtype):
        step_y = 2.0 / H
        step_x = 2.0 / W
        
        start_y = -1.0 + step_y * 0.5
        start_x = -1.0 + step_x * 0.5
        
        grid_y = torch.linspace(start_y, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(start_x, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        
        return grid_y.view(1, H, 1), grid_x.view(1, 1, W)

    def forward(self, flows, images):
        B, L, _, H, W = flows.shape
        _, _, C, _, _ = images.shape
        device = flows.device
        dtype = flows.dtype

        cum_flows = torch.cumsum(flows, dim=1)
        
        rel_flows = cum_flows.unsqueeze(2) - cum_flows.unsqueeze(1) 
        
        flat_rel_flows = rel_flows.view(B * L * L, 2, H, W)
        
        base_grid_y, base_grid_x = self.get_base_grid(B * L * L, H, W, device, dtype)
        
        flow_perm = flat_rel_flows.permute(0, 2, 3, 1)
        
        grid_x = base_grid_x + flow_perm[..., 0]
        grid_y = base_grid_y + flow_perm[..., 1]
        
        grid_x = torch.remainder(grid_x + 1.0, 2.0) - 1.0
        
        final_grid = torch.stack([grid_x, grid_y], dim=-1)

        flat_images = images.view(B, 1, L, C, H, W).expand(-1, L, -1, -1, -1, -1).reshape(B * L * L, C, H, W)

        warped_flat = F.grid_sample(
            flat_images, 
            final_grid, 
            mode=self.mode, 
            padding_mode='zeros', 
            align_corners=False
        )
        
        warped_matrix = warped_flat.view(B, L, L, C, H, W)
        
        causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=dtype)).view(1, L, L, 1, 1, 1)
        
        h_state = (warped_matrix * causal_mask).sum(dim=2)
        
        return h_state

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan(mode=mode)
    return scanner(flows, images)

