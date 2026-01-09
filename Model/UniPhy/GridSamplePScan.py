import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    """
    ICML-Grade Implementation: Split-Kernel Flash Warping
    
    Mathematical Formulation:
    Decomposes the Causal Integration Operator H = L * I into two kernels:
    1. The Dense Off-Diagonal Kernel (Fully vectorized, no masking overhead).
    2. The Sparse Diagonal Kernel (Causal masking applied only locally).
    
    This design maximizes Arithmetic Intensity by removing control flow divergence 
    from the dominant computational path.
    """
    def __init__(self, mode='bilinear', block_size=16):
        super().__init__()
        self.mode = mode
        self.block_size = block_size

    def get_base_grid(self, H, W, device, dtype):
        # Canonical grid cached in a broadcast-ready shape (1, 1, 1, H, W, 2)
        step_y, step_x = 2.0 / H, 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, 1, H, W, 2)

    def _warp_block(self, img_block, flow_diff, base_grid, B, T_len, K_len, C, H, W):
        """Micro-kernel for warping a dense block of pairs."""
        # 1. Geometry: Compute Sampling Grid
        # flow_diff: (B, T_len, K_len, H, W, 2) via broadcasting
        grid = base_grid + flow_diff
        grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0
        
        # 2. Appearance: Prepare Image Features
        # Expand Source: (B, 1, K, C, H, W) -> (B, T, K, C, H, W)
        img_expanded = img_block.unsqueeze(1).expand(-1, T_len, -1, -1, -1, -1)
        
        # 3. Sampling: High-throughput Grid Sample
        # Flatten (B, T, K) into a single batch dimension for the CUDA kernel
        return F.grid_sample(
            img_expanded.reshape(-1, C, H, W),
            grid.reshape(-1, H, W, 2),
            mode=self.mode, 
            padding_mode='zeros', 
            align_corners=False
        ).view(B, T_len, K_len, C, H, W)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype

        # 1. Precompute State (Prefix Sum of Geometry)
        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype).permute(0, 1, 3, 4, 2)
        
        # 2. IO-Aware Tiling: Split into blocks for Cache Locality
        # Views are cheap; no data copy here.
        flow_blocks = cum_flows.split(self.block_size, dim=1)
        img_blocks = images.split(self.block_size, dim=1)
        
        base_grid = self.get_base_grid(H, W, device, dtype)
        output_blocks = []

        # 3. Main Loop: Iterate over Target Blocks (Rows)
        for t_idx, flow_t in enumerate(flow_blocks):
            T_len = flow_t.shape[1]
            flow_t = flow_t.unsqueeze(2) # (B, T, 1, H, W, 2)
            
            # Accumulator for the current time block
            # We start with 0 and functionally add results
            acc_block = 0 

            # --- Phase A: Dense Off-Diagonal Processing (Fast Path) ---
            # Process all strictly past blocks. No masking needed.
            # This loop handles the majority of computations for large L.
            for flow_k, img_k in zip(flow_blocks[:t_idx], img_blocks[:t_idx]):
                K_len = flow_k.shape[1]
                flow_k = flow_k.unsqueeze(1) # (B, 1, K, H, W, 2)
                
                # Compute relative flow and warp
                warped = self._warp_block(img_k, flow_t - flow_k, base_grid, B, T_len, K_len, C, H, W)
                
                # Accumulate: Sum over K dimension
                acc_block = acc_block + warped.sum(dim=2)

            # --- Phase B: Sparse Diagonal Processing (Slow Path) ---
            # Process the current block (where source meets target).
            # Requires causal masking.
            img_k_diag = img_blocks[t_idx]
            flow_k_diag = flow_blocks[t_idx].unsqueeze(1)
            K_len_diag = flow_k_diag.shape[2]
            
            warped_diag = self._warp_block(img_k_diag, flow_t - flow_k_diag, base_grid, B, T_len, K_len_diag, C, H, W)
            
            # Apply Lower Triangular Mask
            # Generate mask on-the-fly to save memory
            mask = torch.tril(torch.ones(T_len, K_len_diag, device=device, dtype=torch.bool))
            warped_diag = warped_diag.masked_fill(~mask.view(1, T_len, K_len_diag, 1, 1, 1), 0)
            
            acc_block = acc_block + warped_diag.sum(dim=2)
            
            # Collect result
            output_blocks.append(acc_block)

        # 4. Final Assembly
        return torch.cat(output_blocks, dim=1)

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    return GridSamplePScan(mode=mode)(flows, images)

