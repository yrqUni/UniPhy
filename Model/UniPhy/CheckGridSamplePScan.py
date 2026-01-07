import torch
import torch.nn.functional as F
from GridSamplePScan import pscan_flow

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

def ref_flow_scan_additive(flows, images, mode='bilinear'):
    B, L, C, H, W = images.shape
    out_images = []
    
    cum_flows = torch.cumsum(flows, dim=1)
    
    for t in range(L):
        h_t = torch.zeros(B, C, H, W, device=flows.device, dtype=flows.dtype)
        
        for k in range(t + 1):
            img_k = images[:, k]
            
            if k == t:
                rel_flow = torch.zeros(B, 2, H, W, device=flows.device, dtype=flows.dtype)
            else:
                rel_flow = cum_flows[:, t] - cum_flows[:, k]
            
            grid = warp_common(rel_flow, B, H, W)
            warped_img_k = F.grid_sample(img_k, grid, mode=mode, padding_mode='zeros', align_corners=False)
            h_t += warped_img_k
            
        out_images.append(h_t)
        
    return torch.stack(out_images, dim=1)

def flow_composition_residual(flow_prev, img_prev, flow_curr, img_curr, mode='bilinear'):
    B, _, H, W = flow_prev.shape
    grid = warp_common(flow_curr, B, H, W)
    flow_sampled = F.grid_sample(flow_prev, grid, mode=mode, padding_mode='zeros', align_corners=False)
    img_sampled = F.grid_sample(img_prev, grid, mode=mode, padding_mode='zeros', align_corners=False)
    return flow_curr + flow_sampled, img_curr + img_sampled

def ref_flow_scan_compositional(flows, images, mode='bilinear'):
    B, L, _, H, W = flows.shape
    out_images = []
    
    curr_flow_acc = flows[:, 0]
    curr_img_acc = images[:, 0]
    out_images.append(curr_img_acc)
    
    for t in range(1, L):
        flow_t = flows[:, t]
        img_t = images[:, t]
        curr_flow_acc, curr_img_acc = flow_composition_residual(
            curr_flow_acc, curr_img_acc, flow_t, img_t, mode=mode
        )
        out_images.append(curr_img_acc)
        
    return torch.stack(out_images, dim=1)

def check_strict_logic(device="cuda"):
    print("=== Strict Logic Check (Comparing Parallel Additive vs Serial Additive) ===")
    
    torch.manual_seed(42)
    B, L, C, H, W = 1, 8, 1, 32, 32
    
    step_y = 2.0 / H
    step_x = 2.0 / W
    shifts_x = torch.randint(-2, 2, (B, L, H, W), device=device).float() * step_x
    shifts_y = torch.randint(-2, 2, (B, L, H, W), device=device).float() * step_y
    flows = torch.stack([shifts_x, shifts_y], dim=2)
    images = torch.randn(B, L, C, H, W, device=device)

    with torch.no_grad():
        out_pscan = pscan_flow(flows, images, mode='nearest')
        out_ref = ref_flow_scan_additive(flows, images, mode='nearest')

    diff = (out_ref - out_pscan).abs().max().item()
    print(f"Max Difference: {diff:.2e}")
    
    if diff < 1e-5:
        print(">> Logic: PASSED (Parallel implementation matches Serial Additive logic)")
    else:
        print(">> Logic: FAILED")

def check_approximation_error(device="cuda"):
    print("\n=== Approximation Error Check (Additive vs Compositional) ===")
    
    torch.manual_seed(42)
    B, L, C, H, W = 2, 8, 4, 32, 32
    
    flows = torch.randn(B, L, 2, H, W, device=device) * 0.05
    images = torch.randn(B, L, C, H, W, device=device)
    
    with torch.no_grad():
        out_pscan = pscan_flow(flows, images, mode='bilinear')
        out_ref = ref_flow_scan_compositional(flows, images, mode='bilinear')

    diff = (out_ref - out_pscan).abs().mean().item()
    print(f"Mean Difference (Approx Error): {diff:.4f}")
    
    flows.requires_grad_(True)
    images.requires_grad_(True)
    loss = pscan_flow(flows, images, mode='bilinear').sum()
    loss.backward()
    
    print(f"Backward Flow Grad Norm: {flows.grad.norm().item():.2e}")
    print(">> Check: DONE")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_strict_logic(device)
    check_approximation_error(device)

