import torch
import torch.nn as nn
import torch.nn.functional as F

def get_base_grid(B, H, W, device, dtype):
    step_y = 2.0 / H
    step_x = 2.0 / W
    start_y = -1.0 + step_y * 0.5
    start_x = -1.0 + step_x * 0.5
    grid_y = torch.linspace(start_y, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
    grid_x = torch.linspace(start_x, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
    return grid_y.view(1, H, 1), grid_x.view(1, 1, W)

def warp_common(flow, B, H, W):
    base_grid_y, base_grid_x = get_base_grid(B, H, W, flow.device, flow.float().dtype)
    flow_perm = flow.permute(0, 2, 3, 1).float()
    final_x = base_grid_x + flow_perm[..., 0]
    final_y = base_grid_y + flow_perm[..., 1]
    final_x = torch.remainder(final_x + 1.0, 2.0) - 1.0
    return torch.stack([final_x, final_y], dim=-1).to(flow.dtype)

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear', channels=None, use_decay=True, use_residual=True, chunk_size=32, window_size=None):
        super().__init__()
        self.mode = mode
        self.use_decay = use_decay
        self.use_residual = use_residual and (channels is not None)
        self.chunk_size = chunk_size
        self.window_size = window_size
        
        self.cached_H = 0
        self.cached_W = 0
        self.register_buffer('base_grid_cache', None, persistent=False)

        if self.use_decay:
            self.decay_log = nn.Parameter(torch.tensor(-2.0))

        if self.use_residual:
            self.res_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 2, kernel_size=1)
            )
            nn.init.zeros_(self.res_conv[-1].weight)
            nn.init.zeros_(self.res_conv[-1].bias)

    def get_cached_grid(self, H, W, device, dtype):
        if self.cached_H != H or self.cached_W != W or self.base_grid_cache is None:
            self.cached_H, self.cached_W = H, W
            step_y, step_x = 2.0 / H, 2.0 / W
            grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=torch.float32)
            grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=torch.float32)
            self.base_grid_cache = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, 1, H, W, 2)
        
        return self.base_grid_cache.to(device)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype
        B_f, L_f, _, H_f, W_f = flows.shape
        is_low_res_flow = (H_f != H) or (W_f != W)

        cum_flows = torch.cumsum(flows.float(), dim=1)
        base_grid_flow = self.get_cached_grid(H_f, W_f, device, dtype)
        
        out_fused = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)
        decay_factor = torch.exp(self.decay_log) if self.use_decay else None

        for t_start in range(0, L, self.chunk_size):
            t_end = min(t_start + self.chunk_size, L)
            curr_t_len = t_end - t_start
            
            acc_chunk = torch.zeros(B, curr_t_len, C, H, W, device=device, dtype=dtype)
            
            min_k = 0
            if self.window_size is not None:
                min_k = max(0, t_start - self.window_size)
            
            for k_start in range(min_k, t_end, self.chunk_size):
                k_end = min(k_start + self.chunk_size, t_end)
                curr_k_len = k_end - k_start

                if self.window_size is not None:
                    if t_start - (k_end - 1) > self.window_size:
                        continue

                flow_t = cum_flows[:, t_start:t_end].unsqueeze(2)
                flow_k = cum_flows[:, k_start:k_end].unsqueeze(1)
                rel_flow = flow_t - flow_k

                if self.use_residual:
                    img_t = images[:, t_start:t_end].unsqueeze(2).expand(-1, -1, curr_k_len, -1, -1, -1)
                    img_k_slice = images[:, k_start:k_end].unsqueeze(1).expand(-1, curr_t_len, -1, -1, -1, -1)
                    
                    feat_diff = torch.cat([img_t, img_k_slice], dim=3)
                    feat_diff_flat = feat_diff.reshape(-1, 2 * C, H, W)

                    if is_low_res_flow:
                        feat_diff_flat = F.interpolate(feat_diff_flat, size=(H_f, W_f), mode='bilinear', align_corners=False)
                    
                    res_flow = self.res_conv(feat_diff_flat).view(B, curr_t_len, curr_k_len, 2, H_f, W_f)
                    rel_flow = rel_flow + res_flow.float()

                grid = base_grid_flow + rel_flow.permute(0, 1, 2, 4, 5, 3)
                grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0

                if is_low_res_flow:
                    grid = grid.reshape(-1, H_f, W_f, 2).permute(0, 3, 1, 2)
                    grid = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)
                    grid = grid.permute(0, 2, 3, 1).view(B, curr_t_len, curr_k_len, H, W, 2)
                
                grid = grid.to(dtype).contiguous()
                
                img_k_input = images[:, k_start:k_end].unsqueeze(1).expand(-1, curr_t_len, -1, -1, -1, -1)
                img_k_input = img_k_input.reshape(-1, C, H, W).contiguous()

                sampled = F.grid_sample(
                    img_k_input,
                    grid.reshape(-1, H, W, 2),
                    mode=self.mode,
                    padding_mode='zeros',
                    align_corners=False
                ).view(B, curr_t_len, curr_k_len, C, H, W)

                t_idx = torch.arange(t_start, t_end, device=device).view(curr_t_len, 1)
                k_idx = torch.arange(k_start, k_end, device=device).view(1, curr_k_len)
                
                mask = (k_idx <= t_idx)
                
                if self.window_size is not None:
                    mask = mask & ((t_idx - k_idx) <= self.window_size)
                
                sampled = sampled * mask.view(1, curr_t_len, curr_k_len, 1, 1, 1)

                if self.use_decay:
                    dist = (t_idx - k_idx).to(dtype)
                    weights = torch.exp(-decay_factor * dist).view(1, curr_t_len, curr_k_len, 1, 1, 1)
                    sampled = sampled * weights

                acc_chunk += sampled.sum(dim=2)

            out_fused[:, t_start:t_end] = acc_chunk

        return out_fused

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    
    C = images.shape[2]
    return GridSamplePScan(mode=mode, channels=C, use_decay=True, use_residual=True, chunk_size=32).to(images.device)(flows, images)

