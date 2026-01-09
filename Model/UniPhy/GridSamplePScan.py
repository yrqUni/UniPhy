import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function

@triton.jit
def fused_pscan_kernel(
    img_ptr, grid_ptr, out_ptr, mask_ptr, decay_dist_ptr,
    B, C, L, H, W, T_chunk, K_chunk,
    stride_img_b, stride_img_l, stride_img_c, stride_img_h, stride_img_w,
    stride_grid_b, stride_grid_t, stride_grid_k, stride_grid_h, stride_grid_w, stride_grid_d,
    stride_out_b, stride_out_t, stride_out_c, stride_out_h, stride_out_w,
    k_start_offset,
    decay_val, use_decay: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    w_idx = offs % W
    tmp = offs // W
    h_idx = tmp % H
    tmp = tmp // H
    t_idx = tmp % T_chunk
    tmp = tmp // T_chunk
    c_idx = tmp % C
    b_idx = tmp // C
    
    mask_valid = b_idx < B
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(K_chunk):
        curr_k_mask = tl.load(mask_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
        
        grid_offset = (b_idx * stride_grid_b + 
                       t_idx * stride_grid_t + 
                       k * stride_grid_k + 
                       h_idx * stride_grid_h + 
                       w_idx * stride_grid_w)
        
        gx = tl.load(grid_ptr + grid_offset + 0 * stride_grid_d, mask=mask_valid, other=0.0)
        gy = tl.load(grid_ptr + grid_offset + 1 * stride_grid_d, mask=mask_valid, other=0.0)
        
        ix = (gx + 1.0) * (W * 0.5) - 0.5
        iy = (gy + 1.0) * (H * 0.5) - 0.5
        
        x0 = tl.floor(ix).to(tl.int32)
        x1 = x0 + 1
        y0 = tl.floor(iy).to(tl.int32)
        y1 = y0 + 1
        
        wa = (x1 - ix) * (y1 - iy)
        wb = (x1 - ix) * (iy - y0)
        wc = (ix - x0) * (y1 - iy)
        wd = (ix - x0) * (iy - y0)
        
        cur_l = k_start_offset + k
        img_base = (b_idx * stride_img_b + 
                    cur_l * stride_img_l + 
                    c_idx * stride_img_c)
        
        check_x0 = (x0 >= 0) & (x0 < W)
        check_x1 = (x1 >= 0) & (x1 < W)
        check_y0 = (y0 >= 0) & (y0 < H)
        check_y1 = (y1 >= 0) & (y1 < H)
        
        val_a = tl.load(img_ptr + img_base + y0 * stride_img_h + x0 * stride_img_w, 
                        mask=mask_valid & check_x0 & check_y0, other=0.0)
        val_b = tl.load(img_ptr + img_base + y1 * stride_img_h + x0 * stride_img_w, 
                        mask=mask_valid & check_x0 & check_y1, other=0.0)
        val_c = tl.load(img_ptr + img_base + y0 * stride_img_h + x1 * stride_img_w, 
                        mask=mask_valid & check_x1 & check_y0, other=0.0)
        val_d = tl.load(img_ptr + img_base + y1 * stride_img_h + x1 * stride_img_w, 
                        mask=mask_valid & check_x1 & check_y1, other=0.0)
        
        interpolated = val_a * wa + val_b * wb + val_c * wc + val_d * wd
        
        weighted = interpolated * curr_k_mask
        
        if use_decay:
            dist = tl.load(decay_dist_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
            w_d = tl.exp(-decay_val * dist)
            weighted = weighted * w_d
            
        acc += weighted

    out_offset = (b_idx * stride_out_b + 
                  t_idx * stride_out_t + 
                  c_idx * stride_out_c + 
                  h_idx * stride_out_h + 
                  w_idx * stride_out_w)
                  
    tl.store(out_ptr + out_offset, acc, mask=mask_valid)


class TritonPScanFunction(Function):
    @staticmethod
    def forward(ctx, images, grid, mask, decay_dist, decay_val, k_start_offset):
        ctx.save_for_backward(images, grid, mask, decay_dist)
        ctx.decay_val = decay_val
        ctx.k_start_offset = k_start_offset
        
        B, L, C, H, W = images.shape
        _, T_chunk, K_chunk, _, _, _ = grid.shape
        
        out = torch.empty((B, T_chunk, C, H, W), device=images.device, dtype=torch.float32)
        
        n_elements = B * C * T_chunk * H * W
        BLOCK_SIZE = 256
        grid_dim = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_pscan_kernel[grid_dim](
            images, grid, out, mask, decay_dist,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            grid.stride(0), grid.stride(1), grid.stride(2), grid.stride(3), grid.stride(4), grid.stride(5),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            k_start_offset,
            decay_val if decay_val is not None else 0.0,
            decay_val is not None,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out.to(images.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        images, grid, mask, decay_dist = ctx.saved_tensors
        decay_val = ctx.decay_val
        k_start_offset = ctx.k_start_offset
        
        B, L, C, H, W = images.shape
        _, T_chunk, K_chunk, _, _, _ = grid.shape
        
        grad_images = torch.zeros_like(images) if images.requires_grad else None
        grad_grid = torch.zeros_like(grid) if grid.requires_grad else None
        
        if grad_images is None and grad_grid is None:
            return None, None, None, None, None, None

        with torch.enable_grad():
            for k in range(K_chunk):
                img_idx = k_start_offset + k
                if img_idx >= L: continue
                
                curr_img = images[:, img_idx].detach()
                curr_img.requires_grad_(True)
                
                curr_grid = grid[:, :, k].detach()
                curr_grid.requires_grad_(True)
                
                img_expanded = curr_img.unsqueeze(1).expand(-1, T_chunk, -1, -1, -1)
                img_reshaped = img_expanded.reshape(B * T_chunk, C, H, W)
                
                grid_reshaped = curr_grid.reshape(B * T_chunk, H, W, 2)
                
                sampled = F.grid_sample(
                    img_reshaped, 
                    grid_reshaped, 
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=False
                )
                sampled = sampled.view(B, T_chunk, C, H, W)
                
                curr_mask = mask[:, k].view(1, T_chunk, 1, 1, 1)
                weighted = sampled * curr_mask
                
                if decay_val is not None:
                    curr_dist = decay_dist[:, k].view(1, T_chunk, 1, 1, 1)
                    w_d = torch.exp(-decay_val * curr_dist)
                    weighted = weighted * w_d
                
                weighted.backward(grad_output, retain_graph=False)
                
                if grad_images is not None:
                    grad_images[:, img_idx] += curr_img.grad
                
                if grad_grid is not None:
                    grad_grid[:, :, k] += curr_grid.grad
        
        return grad_images, grad_grid, None, None, None, None


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
        decay_val = torch.exp(self.decay_log).item() if self.use_decay else None

        for t_start in range(0, L, self.chunk_size):
            t_end = min(t_start + self.chunk_size, L)
            curr_t_len = t_end - t_start
            
            flow_t = cum_flows[:, t_start:t_end].unsqueeze(2)
            
            img_t_slice = None
            if self.use_residual:
                img_t_slice = images[:, t_start:t_end].unsqueeze(2)

            t_idx = torch.arange(t_start, t_end, device=device).view(curr_t_len, 1)

            min_k = 0
            if self.window_size is not None:
                min_k = max(0, t_start - self.window_size)
            
            for k_start in range(min_k, t_end, self.chunk_size):
                k_end = min(k_start + self.chunk_size, t_end)
                curr_k_len = k_end - k_start

                if self.window_size is not None:
                    if t_start - (k_end - 1) > self.window_size:
                        continue

                flow_k = cum_flows[:, k_start:k_end].unsqueeze(1)
                rel_flow = flow_t - flow_k

                if self.use_residual:
                    img_t = img_t_slice.expand(-1, -1, curr_k_len, -1, -1, -1)
                    img_k = images[:, k_start:k_end].unsqueeze(1).expand(-1, curr_t_len, -1, -1, -1, -1)
                    
                    feat_diff = torch.cat([img_t, img_k], dim=3)
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
                
                grid = grid.contiguous()
                
                k_idx = torch.arange(k_start, k_end, device=device).view(1, curr_k_len)
                mask = (k_idx <= t_idx)
                
                if self.window_size is not None:
                    mask = mask & ((t_idx - k_idx) <= self.window_size)
                
                mask = mask.float().contiguous()
                dist = (t_idx - k_idx).float().contiguous()

                acc_chunk = TritonPScanFunction.apply(images, grid, mask, dist, decay_val, k_start)
                out_fused[:, t_start:t_end] += acc_chunk

        return out_fused

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    
    C = images.shape[2]
    return GridSamplePScan(mode=mode, channels=C, use_decay=True, use_residual=True, chunk_size=32).to(images.device)(flows, images)

