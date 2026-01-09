import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear', block_size=16):
        super().__init__()
        self.mode = mode
        self.block_size = block_size

    def get_base_grid(self, H, W, device, dtype):
        step_y, step_x = 2.0 / H, 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, 1, H, W, 2)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype

        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype).permute(0, 1, 3, 4, 2)
        flow_blocks = cum_flows.split(self.block_size, dim=1)
        img_blocks = images.split(self.block_size, dim=1)
        
        base_grid = self.get_base_grid(H, W, device, dtype)
        h_blocks = []

        for t_idx, flow_t in enumerate(flow_blocks):
            T_len = flow_t.shape[1]
            flow_t = flow_t.unsqueeze(2)
            
            acc = 0

            for k_idx, (flow_k, img_k) in enumerate(zip(flow_blocks[:t_idx+1], img_blocks[:t_idx+1])):
                K_len = flow_k.shape[1]
                flow_k = flow_k.unsqueeze(1)

                grid = base_grid + (flow_t - flow_k)
                grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0

                img_expanded = img_k.unsqueeze(1).expand(-1, T_len, -1, -1, -1, -1)
                
                warped = F.grid_sample(
                    img_expanded.reshape(-1, C, H, W),
                    grid.reshape(-1, H, W, 2),
                    mode=self.mode, 
                    padding_mode='zeros', 
                    align_corners=False
                ).view(B, T_len, K_len, C, H, W)

                if t_idx == k_idx:
                    mask = torch.tril(torch.ones(T_len, K_len, device=device, dtype=torch.bool))
                    warped = warped.masked_fill(~mask.view(1, T_len, K_len, 1, 1, 1), 0)

                acc = acc + warped.sum(dim=2)
            
            h_blocks.append(acc)

        return torch.cat(h_blocks, dim=1)

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    return GridSamplePScan(mode=mode)(flows, images)

