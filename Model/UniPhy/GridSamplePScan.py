import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        if mode not in ['bilinear', 'nearest']:
            raise ValueError(f"mode must be 'bilinear' or 'nearest', got {mode}")
        self.mode = mode

    def forward(self, flows, images):
        B, L, _, H, W = flows.shape
        _, _, C, _, _ = images.shape
        device = flows.device
        dtype = flows.dtype

        # 1. Precompute Cumulative Flow (B, L, 2, H, W)
        # Use FP32 for accumulation precision, then cast back
        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype)

        # 2. Generate Causal Indices (k <= t)
        # N_pairs = L * (L + 1) / 2. This reduces memory/compute by ~50% vs L*L
        # k_idx: source time index, t_idx: target time index
        k_idx, t_idx = torch.triu_indices(L, L, offset=0, device=device)
        N_pairs = k_idx.numel()

        # 3. Prepare Batch Offsets
        # We need to process all batches for these pairs.
        # Total effective batch size: B * N_pairs
        batch_idx = torch.arange(B, device=device)[:, None].expand(B, N_pairs).reshape(-1)
        k_idx_flat = k_idx[None, :].expand(B, N_pairs).reshape(-1)
        t_idx_flat = t_idx[None, :].expand(B, N_pairs).reshape(-1)
        
        # 4. Gather Flows and Compute Relative Flow
        # flow_{k->t} = cum_flows[t] - cum_flows[k]
        # We gather (B * N_pairs, 2, H, W) directly
        flows_t = cum_flows[batch_idx, t_idx_flat]
        flows_k = cum_flows[batch_idx, k_idx_flat]
        flat_rel_flows = flows_t - flows_k

        # 5. Generate Grid on the fly
        # Grid generation is cheap compared to memory bandwidth of storing it
        step_y = 2.0 / H
        step_x = 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        # (N, H, W, 2)
        base_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).unsqueeze(0)
        
        # Apply Flow to Grid
        final_grid = base_grid + flat_rel_flows.permute(0, 2, 3, 1)
        # Wrap Longitude (X-axis)
        final_grid[..., 0] = torch.remainder(final_grid[..., 0] + 1.0, 2.0) - 1.0

        # 6. Gather Source Images
        # Extract images at time k: (B * N_pairs, C, H, W)
        flat_images = images[batch_idx, k_idx_flat]

        # 7. Grid Sample (The Heavy Lifter)
        # Only computing valid causal pairs
        warped_flat = F.grid_sample(
            flat_images, 
            final_grid, 
            mode=self.mode, 
            padding_mode='zeros', 
            align_corners=False
        )

        # 8. Scatter Accumulate Results
        # Accumulate warped_flat into h_state based on target time index t
        # Output: (B, L, C, H, W)
        h_state = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)
        
        # We need to reshape warped_flat to map back to (B, N_pairs, ...) for index_add
        # But index_add_ only works on a specific dimension. 
        # Easier strategy: Reshape h_state to (B*L, C, H, W) and index_add_ using flat indices
        
        target_flat_indices = batch_idx * L + t_idx_flat # Map (b, t) to global index
        
        h_state_flat = h_state.view(B * L, C, H, W)
        h_state_flat.index_add_(0, target_flat_indices, warped_flat)
        
        return h_state_flat.view(B, L, C, H, W)

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    scanner = GridSamplePScan(mode=mode)
    return scanner(flows, images)

