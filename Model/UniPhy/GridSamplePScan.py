import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def get_base_grid(self, B, H, W, device, dtype):
        step_y, step_x = 2.0 / H, 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, 1, H, W, 2)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype

        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype)
        base_grid = self.get_base_grid(B, H, W, device, dtype)

        out_fused = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)

        for t in range(L):
            flow_t = cum_flows[:, t:t+1]
            flow_k = cum_flows[:, :t+1]
            
            rel_flow = flow_t - flow_k

            grid = base_grid + rel_flow.permute(0, 1, 3, 4, 2)
            grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0

            img_k = images[:, :t+1]

            sampled = F.grid_sample(
                img_k.reshape(-1, C, H, W),
                grid.reshape(-1, H, W, 2),
                mode=self.mode,
                padding_mode='zeros',
                align_corners=False
            )

            out_fused[:, t] = sampled.view(B, t + 1, C, H, W).sum(dim=1)

        return out_fused

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    return GridSamplePScan(mode=mode)(flows, images)

