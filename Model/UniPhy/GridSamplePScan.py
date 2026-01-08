import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear', chunk_size=1024):
        super().__init__()
        if mode not in ['bilinear', 'nearest']:
            raise ValueError(f"mode must be 'bilinear' or 'nearest', got {mode}")
        self.mode = mode
        self.chunk_size = chunk_size

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype

        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype)

        k_idx, t_idx = torch.triu_indices(L, L, offset=0, device=device)
        N_pairs = k_idx.numel()

        step_y = 2.0 / H
        step_x = 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        base_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).unsqueeze(0)

        h_state = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)
        h_state_flat = h_state.view(B * L, C, H, W)

        for start in range(0, N_pairs, self.chunk_size):
            end = min(start + self.chunk_size, N_pairs)
            current_pairs = end - start
            
            k_chunk = k_idx[start:end]
            t_chunk = t_idx[start:end]

            batch_idx = torch.arange(B, device=device)[:, None].expand(B, current_pairs).reshape(-1)
            k_flat = k_chunk[None, :].expand(B, current_pairs).reshape(-1)
            t_flat = t_chunk[None, :].expand(B, current_pairs).reshape(-1)

            sub_flows_t = cum_flows[batch_idx, t_flat]
            sub_flows_k = cum_flows[batch_idx, k_flat]
            sub_rel_flows = sub_flows_t - sub_flows_k

            sub_grid = base_grid + sub_rel_flows.permute(0, 2, 3, 1)
            sub_grid[..., 0] = torch.remainder(sub_grid[..., 0] + 1.0, 2.0) - 1.0

            sub_images = images[batch_idx, k_flat]

            warped_chunk = F.grid_sample(
                sub_images, 
                sub_grid, 
                mode=self.mode, 
                padding_mode='zeros', 
                align_corners=False
            )

            target_flat_indices = batch_idx * L + t_flat
            h_state_flat.index_add_(0, target_flat_indices, warped_chunk)

        return h_state_flat.view(B, L, C, H, W)

def pscan_flow(flows, images, mode='bilinear', chunk_size=4096):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan(mode=mode, chunk_size=chunk_size)
    return scanner(flows, images)

