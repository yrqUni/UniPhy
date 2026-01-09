import torch
import torch.nn as nn
import torch.nn.functional as F

class GridSamplePScan(nn.Module):
    def __init__(self, mode='bilinear', channels=None, use_decay=True, use_residual=True):
        super().__init__()
        self.mode = mode
        self.use_decay = use_decay
        self.use_residual = use_residual and (channels is not None)

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

    def get_base_grid(self, B, H, W, device, dtype):
        step_y, step_x = 2.0 / H, 2.0 / W
        grid_y = torch.linspace(-1.0 + step_y * 0.5, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0 + step_x * 0.5, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1).view(1, 1, 1, H, W, 2)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype
        
        B_f, L_f, _, H_f, W_f = flows.shape
        is_low_res_flow = (H_f != H) or (W_f != W)

        cum_flows = torch.cumsum(flows.float(), dim=1).to(dtype)
        base_grid_flow = self.get_base_grid(B, H_f, W_f, device, dtype)

        out_fused = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)

        decay_factor = torch.exp(self.decay_log) if self.use_decay else None

        for t in range(L):
            k_len = t + 1
            
            flow_t = cum_flows[:, t:t+1]
            flow_k = cum_flows[:, :k_len]
            rel_flow = flow_t - flow_k 

            if self.use_residual:
                img_t = images[:, t:t+1]
                img_k = images[:, :k_len]
                img_t_expanded = img_t.expand(-1, k_len, -1, -1, -1)
                
                feat_diff = torch.cat([img_t_expanded, img_k], dim=2)
                feat_diff_flat = feat_diff.reshape(B * k_len, 2 * C, H, W)
                
                if is_low_res_flow:
                    feat_diff_flat = F.interpolate(feat_diff_flat, size=(H_f, W_f), mode='bilinear', align_corners=False)
                
                res_flow = self.res_conv(feat_diff_flat).view(B, k_len, 2, H_f, W_f)
                rel_flow = rel_flow + res_flow

            grid = base_grid_flow + rel_flow.permute(0, 1, 3, 4, 2)
            grid[..., 0] = torch.remainder(grid[..., 0] + 1.0, 2.0) - 1.0

            if is_low_res_flow:
                grid = grid.view(B * k_len, H_f, W_f, 2).permute(0, 3, 1, 2)
                grid = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)
                grid = grid.permute(0, 2, 3, 1).view(B, k_len, H, W, 2)

            img_k = images[:, :k_len]
            sampled = F.grid_sample(
                img_k.reshape(-1, C, H, W),
                grid.reshape(-1, H, W, 2),
                mode=self.mode,
                padding_mode='zeros',
                align_corners=False
            ).view(B, k_len, C, H, W)

            if self.use_decay:
                times = torch.arange(k_len, device=device, dtype=dtype)
                weights = torch.exp(-decay_factor * (t - times)).view(1, k_len, 1, 1, 1)
                sampled = sampled * weights

            out_fused[:, t] = sampled.sum(dim=1)

        return out_fused

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    
    C = images.shape[2]
    return GridSamplePScan(mode=mode, channels=C, use_decay=True, use_residual=True).to(images.device)(flows, images)

