import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function

def get_base_grid(B, H, W, device, dtype):
    step_y = 2.0 / H
    step_x = 2.0 / W
    start_y = -1.0 + step_y * 0.5
    start_x = -1.0 + step_x * 0.5
    grid_y = torch.linspace(start_y, 1.0 - step_y * 0.5, H, device=device, dtype=dtype)
    grid_x = torch.linspace(start_x, 1.0 - step_x * 0.5, W, device=device, dtype=dtype)
    return grid_y.view(1, H, 1), grid_x.view(1, 1, W)

def warp_common(flow, B, H, W):
    base_grid_y, base_grid_x = get_base_grid(B, H, W, flow.device, flow.dtype)
    flow_perm = flow.permute(0, 2, 3, 1)
    final_x = base_grid_x + flow_perm[..., 0]
    final_y = base_grid_y + flow_perm[..., 1]
    final_x = torch.remainder(final_x + 1.0, 2.0) - 1.0
    return torch.stack([final_x, final_y], dim=-1)

@triton.jit
def fused_pscan_forward_kernel_2d(
    img_ptr, cum_flow_ptr, res_flow_ptr, out_ptr, mask_ptr, decay_dist_ptr,
    B, C, L, H, W, T_chunk, K_chunk,
    stride_img_b, stride_img_l, stride_img_c, stride_img_h, stride_img_w,
    stride_flow_b, stride_flow_l, stride_flow_c, stride_flow_h, stride_flow_w,
    stride_res_b, stride_res_t, stride_res_k, stride_res_c, stride_res_h, stride_res_w,
    stride_out_b, stride_out_t, stride_out_c, stride_out_h, stride_out_w,
    k_start_offset, t_start_offset,
    decay_val, use_decay: tl.constexpr, use_res_flow: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    num_bh = tl.cdiv(H, BLOCK_H)
    num_bw = tl.cdiv(W, BLOCK_W)
    
    off_h = (pid_s // num_bw) * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = (pid_s % num_bw) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    t_idx = pid_b % T_chunk
    tmp = pid_b // T_chunk
    c_idx = tmp % C
    b_idx = tmp // C

    mask_valid = (b_idx < B)
    mask_h = (off_h < H)
    mask_w = (off_w < W)
    mask_hw = mask_h[:, None] & mask_w[None, :]

    base_y = (off_h[:, None].to(tl.float32) * (2.0 / H)) - 1.0 + (1.0 / H)
    base_x = (off_w[None, :].to(tl.float32) * (2.0 / W)) - 1.0 + (1.0 / W)

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    
    t_abs = t_start_offset + t_idx
    k_end = t_idx + 1 if (k_start_offset == t_start_offset) else K_chunk
    k_end = tl.minimum(k_end, K_chunk)

    flow_t_ptr = cum_flow_ptr + b_idx * stride_flow_b + t_abs * stride_flow_l
    img_base_ptr = img_ptr + b_idx * stride_img_b + c_idx * stride_img_c

    for k in range(k_end):
        k_abs = k_start_offset + k
        curr_k_mask = tl.load(mask_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
        
        if curr_k_mask != 0.0:
            flow_k_ptr = cum_flow_ptr + b_idx * stride_flow_b + k_abs * stride_flow_l
            
            off_flow_x = off_w[None, :] * stride_flow_w + off_h[:, None] * stride_flow_h
            
            fx_t = tl.load(flow_t_ptr + 0 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            fy_t = tl.load(flow_t_ptr + 1 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            
            fx_k = tl.load(flow_k_ptr + 0 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            fy_k = tl.load(flow_k_ptr + 1 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)

            res_x = 0.0
            res_y = 0.0
            if use_res_flow:
                res_base = (b_idx * stride_res_b + t_idx * stride_res_t + k * stride_res_k)
                res_x = tl.load(res_flow_ptr + res_base + 0 * stride_res_c + off_flow_x, mask=mask_hw, other=0.0)
                res_y = tl.load(res_flow_ptr + res_base + 1 * stride_res_c + off_flow_x, mask=mask_hw, other=0.0)

            gx = base_x + (fx_t - fx_k) + res_x
            gy = base_y + (fy_t - fy_k) + res_y

            val_x = gx + 1.0
            val_x = val_x - 2.0 * tl.floor(val_x * 0.5)
            gx = val_x - 1.0

            ix = (gx + 1.0) * (W * 0.5) - 0.5
            iy = (gy + 1.0) * (H * 0.5) - 0.5

            x0 = tl.floor(ix).to(tl.int32)
            y0 = tl.floor(iy).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            wa = (x1 - ix) * (y1 - iy)
            wb = (x1 - ix) * (iy - y0)
            wc = (ix - x0) * (y1 - iy)
            wd = (ix - x0) * (iy - y0)

            img_k_ptr = img_base_ptr + k_abs * stride_img_l

            c_x0 = (x0 >= 0) & (x0 < W)
            c_x1 = (x1 >= 0) & (x1 < W)
            c_y0 = (y0 >= 0) & (y0 < H)
            c_y1 = (y1 >= 0) & (y1 < H)

            val_a = tl.load(img_k_ptr + y0 * stride_img_h + x0 * stride_img_w, mask=mask_valid & c_x0 & c_y0 & mask_hw, other=0.0)
            val_b = tl.load(img_k_ptr + y1 * stride_img_h + x0 * stride_img_w, mask=mask_valid & c_x0 & c_y1 & mask_hw, other=0.0)
            val_c = tl.load(img_k_ptr + y0 * stride_img_h + x1 * stride_img_w, mask=mask_valid & c_x1 & c_y0 & mask_hw, other=0.0)
            val_d = tl.load(img_k_ptr + y1 * stride_img_h + x1 * stride_img_w, mask=mask_valid & c_x1 & c_y1 & mask_hw, other=0.0)

            weight = curr_k_mask
            if use_decay:
                dist = tl.load(decay_dist_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
                weight = weight * tl.exp(-decay_val * dist)

            acc += (val_a * wa + val_b * wb + val_c * wc + val_d * wd) * weight

    out_offset = (b_idx * stride_out_b + t_idx * stride_out_t + c_idx * stride_out_c + 
                  off_h[:, None] * stride_out_h + off_w[None, :] * stride_out_w)
    tl.store(out_ptr + out_offset, acc, mask=mask_valid & mask_hw)

@triton.jit
def fused_pscan_backward_kernel_2d(
    grad_out_ptr, img_ptr, cum_flow_ptr, res_flow_ptr, mask_ptr, decay_dist_ptr,
    grad_img_ptr, grad_flow_ptr, grad_res_ptr,
    B, C, L, H, W, T_chunk, K_chunk,
    stride_img_b, stride_img_l, stride_img_c, stride_img_h, stride_img_w,
    stride_flow_b, stride_flow_l, stride_flow_c, stride_flow_h, stride_flow_w,
    stride_res_b, stride_res_t, stride_res_k, stride_res_c, stride_res_h, stride_res_w,
    stride_out_b, stride_out_t, stride_out_c, stride_out_h, stride_out_w,
    k_start_offset, t_start_offset,
    decay_val, use_decay: tl.constexpr, use_res_flow: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    num_bh = tl.cdiv(H, BLOCK_H)
    num_bw = tl.cdiv(W, BLOCK_W)
    
    off_h = (pid_s // num_bw) * BLOCK_H + tl.arange(0, BLOCK_H)
    off_w = (pid_s % num_bw) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    t_idx = pid_b % T_chunk
    tmp = pid_b // T_chunk
    c_idx = tmp % C
    b_idx = tmp // C

    mask_valid = (b_idx < B)
    mask_h = (off_h < H)
    mask_w = (off_w < W)
    mask_hw = mask_h[:, None] & mask_w[None, :]

    base_y = (off_h[:, None].to(tl.float32) * (2.0 / H)) - 1.0 + (1.0 / H)
    base_x = (off_w[None, :].to(tl.float32) * (2.0 / W)) - 1.0 + (1.0 / W)
    
    grad_out_off = (b_idx * stride_out_b + t_idx * stride_out_t + c_idx * stride_out_c + 
                    off_h[:, None] * stride_out_h + off_w[None, :] * stride_out_w)
    grad_out_val = tl.load(grad_out_ptr + grad_out_off, mask=mask_valid & mask_hw, other=0.0)

    t_abs = t_start_offset + t_idx
    k_end = t_idx + 1 if (k_start_offset == t_start_offset) else K_chunk
    k_end = tl.minimum(k_end, K_chunk)

    flow_t_ptr = cum_flow_ptr + b_idx * stride_flow_b + t_abs * stride_flow_l
    img_base_ptr = img_ptr + b_idx * stride_img_b + c_idx * stride_img_c
    grad_img_base_ptr = grad_img_ptr + b_idx * stride_img_b + c_idx * stride_img_c
    
    if grad_flow_ptr is not None:
        grad_flow_t_ptr = grad_flow_ptr + b_idx * stride_flow_b + t_abs * stride_flow_l

    for k in range(k_end):
        k_abs = k_start_offset + k
        curr_k_mask = tl.load(mask_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
        
        if curr_k_mask != 0.0:
            flow_k_ptr = cum_flow_ptr + b_idx * stride_flow_b + k_abs * stride_flow_l
            
            off_flow_x = off_w[None, :] * stride_flow_w + off_h[:, None] * stride_flow_h
            
            fx_t = tl.load(flow_t_ptr + 0 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            fy_t = tl.load(flow_t_ptr + 1 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            fx_k = tl.load(flow_k_ptr + 0 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)
            fy_k = tl.load(flow_k_ptr + 1 * stride_flow_c + off_flow_x, mask=mask_hw, other=0.0)

            res_x = 0.0
            res_y = 0.0
            if use_res_flow:
                res_base = (b_idx * stride_res_b + t_idx * stride_res_t + k * stride_res_k)
                res_x = tl.load(res_flow_ptr + res_base + 0 * stride_res_c + off_flow_x, mask=mask_hw, other=0.0)
                res_y = tl.load(res_flow_ptr + res_base + 1 * stride_res_c + off_flow_x, mask=mask_hw, other=0.0)

            gx = base_x + (fx_t - fx_k) + res_x
            gy = base_y + (fy_t - fy_k) + res_y

            val_x = gx + 1.0
            val_x = val_x - 2.0 * tl.floor(val_x * 0.5)
            gx = val_x - 1.0

            ix = (gx + 1.0) * (W * 0.5) - 0.5
            iy = (gy + 1.0) * (H * 0.5) - 0.5

            x0 = tl.floor(ix).to(tl.int32)
            y0 = tl.floor(iy).to(tl.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            wa = (x1 - ix) * (y1 - iy)
            wb = (x1 - ix) * (iy - y0)
            wc = (ix - x0) * (y1 - iy)
            wd = (ix - x0) * (iy - y0)

            img_k_ptr = img_base_ptr + k_abs * stride_img_l
            
            c_x0 = (x0 >= 0) & (x0 < W)
            c_x1 = (x1 >= 0) & (x1 < W)
            c_y0 = (y0 >= 0) & (y0 < H)
            c_y1 = (y1 >= 0) & (y1 < H)

            val_a = tl.load(img_k_ptr + y0 * stride_img_h + x0 * stride_img_w, mask=mask_valid & c_x0 & c_y0 & mask_hw, other=0.0)
            val_b = tl.load(img_k_ptr + y1 * stride_img_h + x0 * stride_img_w, mask=mask_valid & c_x0 & c_y1 & mask_hw, other=0.0)
            val_c = tl.load(img_k_ptr + y0 * stride_img_h + x1 * stride_img_w, mask=mask_valid & c_x1 & c_y0 & mask_hw, other=0.0)
            val_d = tl.load(img_k_ptr + y1 * stride_img_h + x1 * stride_img_w, mask=mask_valid & c_x1 & c_y1 & mask_hw, other=0.0)

            weight = curr_k_mask
            if use_decay:
                dist = tl.load(decay_dist_ptr + t_idx * K_chunk + k, mask=mask_valid, other=0.0)
                weight = weight * tl.exp(-decay_val * dist)

            grad_curr = grad_out_val * weight
            
            is_significant = tl.abs(grad_curr) > 1e-10
            mask_ops = mask_valid & mask_hw & is_significant

            dwa_dx = -(y1 - iy)
            dwa_dy = -(x1 - ix)
            dwb_dx = -(iy - y0)
            dwb_dy = (x1 - ix)
            dwc_dx = (y1 - iy)
            dwc_dy = -(ix - x0)
            dwd_dx = (iy - y0)
            dwd_dy = (ix - x0)

            dval_dx = val_a * dwa_dx + val_b * dwb_dx + val_c * dwc_dx + val_d * dwd_dx
            dval_dy = val_a * dwa_dy + val_b * dwb_dy + val_c * dwc_dy + val_d * dwd_dy
            
            grad_gx = grad_curr * dval_dx * (W * 0.5)
            grad_gy = grad_curr * dval_dy * (H * 0.5)
            
            grad_a = grad_curr * wa
            grad_b = grad_curr * wb
            grad_c = grad_curr * wc
            grad_d = grad_curr * wd

            grad_img_k_ptr = grad_img_base_ptr + k_abs * stride_img_l
            # [TOPOLOGY FIX] Recalculate coordinates for gradients
            x0_r = (x0 % W + W) % W
            x1_r = (x1 % W + W) % W
            y0_c = tl.maximum(0, tl.minimum(H - 1, y0))
            y1_c = tl.maximum(0, tl.minimum(H - 1, y1))

            tl.atomic_add(grad_img_k_ptr + y0_c * stride_img_h + x0_r * stride_img_w, grad_a, mask=mask_ops)
            tl.atomic_add(grad_img_k_ptr + y1_c * stride_img_h + x0_r * stride_img_w, grad_b, mask=mask_ops)
            tl.atomic_add(grad_img_k_ptr + y0_c * stride_img_h + x1_r * stride_img_w, grad_c, mask=mask_ops)
            tl.atomic_add(grad_img_k_ptr + y1_c * stride_img_h + x1_r * stride_img_w, grad_d, mask=mask_ops)
            
            if grad_flow_ptr is not None:
                grad_flow_k_ptr = grad_flow_ptr + b_idx * stride_flow_b + k_abs * stride_flow_l
                
                tl.atomic_add(grad_flow_t_ptr + 0 * stride_flow_c + off_flow_x, grad_gx, mask=mask_ops)
                tl.atomic_add(grad_flow_t_ptr + 1 * stride_flow_c + off_flow_x, grad_gy, mask=mask_ops)
                
                tl.atomic_add(grad_flow_k_ptr + 0 * stride_flow_c + off_flow_x, -grad_gx, mask=mask_ops)
                tl.atomic_add(grad_flow_k_ptr + 1 * stride_flow_c + off_flow_x, -grad_gy, mask=mask_ops)
            
            if grad_res_ptr is not None and use_res_flow:
                res_base_grad = (b_idx * stride_res_b + t_idx * stride_res_t + k * stride_res_k)
                tl.atomic_add(grad_res_ptr + res_base_grad + 0 * stride_res_c + off_flow_x, grad_gx, mask=mask_ops)
                tl.atomic_add(grad_res_ptr + res_base_grad + 1 * stride_res_c + off_flow_x, grad_gy, mask=mask_ops)

class TritonFunction(Function):
    @staticmethod
    def forward(ctx, images, cum_flows, res_flows, mask, decay_dist, decay_val, k_start_offset, t_start_offset):
        ctx.save_for_backward(images, cum_flows, res_flows, mask, decay_dist)
        ctx.decay_val = decay_val
        ctx.k_start_offset = k_start_offset
        ctx.t_start_offset = t_start_offset
        
        B, L, C, H, W = images.shape
        T_chunk, K_chunk = mask.shape[1], mask.shape[2]
        
        out = torch.empty((B, T_chunk, C, H, W), device=images.device, dtype=torch.float32)
        
        BLOCK_H = 16
        BLOCK_W = 16
        grid_dim = (triton.cdiv(H * W, BLOCK_H * BLOCK_W), B * T_chunk * C)
        
        use_res_flow = res_flows is not None
        if use_res_flow:
            s = res_flows.stride()
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]
        else:
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0

        fused_pscan_forward_kernel_2d[grid_dim](
            images, cum_flows, res_flows, out, mask, decay_dist,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            k_start_offset, t_start_offset,
            decay_val if decay_val is not None else 0.0,
            decay_val is not None, use_res_flow,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        
        return out.to(images.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        images, cum_flows, res_flows, mask, decay_dist = ctx.saved_tensors
        decay_val = ctx.decay_val
        k_start_offset = ctx.k_start_offset
        t_start_offset = ctx.t_start_offset
        
        B, L, C, H, W = images.shape
        T_chunk, K_chunk = mask.shape[1], mask.shape[2]
        
        grad_images = torch.zeros_like(images) if images.requires_grad else None
        grad_cum_flows = torch.zeros_like(cum_flows) if cum_flows.requires_grad else None
        grad_res_flows = torch.zeros_like(res_flows) if (res_flows is not None and res_flows.requires_grad) else None
        
        if grad_images is None and grad_cum_flows is None and grad_res_flows is None:
            return None, None, None, None, None, None, None, None

        BLOCK_H = 16
        BLOCK_W = 16
        grid_dim = (triton.cdiv(H * W, BLOCK_H * BLOCK_W), B * T_chunk * C)
        
        grad_output = grad_output.contiguous()
        use_res_flow = res_flows is not None
        if use_res_flow:
            s = res_flows.stride()
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = s[0], 0, s[1], s[2], s[3], s[4]
        else:
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w = 0, 0, 0, 0, 0, 0

        fused_pscan_backward_kernel_2d[grid_dim](
            grad_output, images, cum_flows, res_flows, mask, decay_dist,
            grad_images, grad_cum_flows, grad_res_flows,
            B, C, L, H, W, T_chunk, K_chunk,
            images.stride(0), images.stride(1), images.stride(2), images.stride(3), images.stride(4),
            cum_flows.stride(0), cum_flows.stride(1), cum_flows.stride(2), cum_flows.stride(3), cum_flows.stride(4),
            rs_b, rs_t, rs_k, rs_c, rs_h, rs_w,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3), grad_output.stride(4),
            k_start_offset, t_start_offset,
            decay_val if decay_val is not None else 0.0,
            decay_val is not None, use_res_flow,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        
        return grad_images, grad_cum_flows, grad_res_flows, None, None, None, None, None

class GridSample(nn.Module):
    def __init__(self, mode='bilinear', channels=None, use_decay=True, use_residual=True, chunk_size=32, window_size=None):
        super().__init__()
        self.mode = mode
        self.use_decay = use_decay
        self.use_residual = use_residual and (channels is not None)
        self.chunk_size = chunk_size
        self.window_size = window_size
        
        if self.use_decay:
            self.decay_log = nn.Parameter(torch.tensor(-2.0), requires_grad=False)

        if self.use_residual:
            self.res_conv = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, 2, kernel_size=1)
            )
            nn.init.zeros_(self.res_conv[-1].weight)
            nn.init.zeros_(self.res_conv[-1].bias)

    def forward(self, flows, images):
        B, L, C, H, W = images.shape
        device = flows.device
        dtype = flows.dtype
        B_f, L_f, _, H_f, W_f = flows.shape
        
        cum_flows = torch.cumsum(flows.float(), dim=1)
        
        out_fused = torch.zeros(B, L, C, H, W, device=device, dtype=dtype)
        decay_val = None 

        for t_start in range(0, L, self.chunk_size):
            t_end = min(t_start + self.chunk_size, L)
            curr_t_len = t_end - t_start
            
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
                
                res_flow_chunk = None
                if self.use_residual:
                    img_chunk = images[:, k_start:k_end]
                    b_chk, t_chk, c, h, w = img_chunk.shape
                    img_flat = img_chunk.reshape(b_chk * t_chk, c, h, w)
                    res_flat = self.res_conv(img_flat)
                    res_flow_chunk = res_flat.view(b_chk, t_chk, 2, h, w)

                k_idx = torch.arange(k_start, k_end, device=device).view(1, curr_k_len)
                mask = (k_idx <= t_idx)
                
                if self.window_size is not None:
                    mask = mask & ((t_idx - k_idx) <= self.window_size)
                
                mask = mask.float().contiguous().unsqueeze(0).expand(B, -1, -1)
                dist = (t_idx - k_idx).float().contiguous().unsqueeze(0).expand(B, -1, -1)

                acc_chunk = TritonFunction.apply(images, cum_flows, res_flow_chunk, mask, dist, decay_val, k_start, t_start)
                out_fused[:, t_start:t_end] += acc_chunk

        return out_fused

def pscan_flow(flows, images, mode='bilinear'):
    if flows.size(-1) == 2:
        flows = flows.permute(0, 1, 4, 2, 3)
    
    C = images.shape[2]
    return GridSample(mode=mode, channels=C, use_decay=True, use_residual=True, chunk_size=32).to(images.device)(flows, images)

