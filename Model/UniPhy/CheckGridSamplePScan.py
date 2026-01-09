import torch
import torch.nn.functional as F
from GridSamplePScan import GridSamplePScan

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

def check_strict_logic(device="cuda"):
    print("=== Strict Logic Check (Core Additive Logic) ===")

    torch.manual_seed(42)
    B, L, C, H, W = 1, 8, 1, 32, 32

    step_y = 2.0 / H
    step_x = 2.0 / W
    shifts_x = torch.randint(-2, 2, (B, L, H, W), device=device).float() * step_x
    shifts_y = torch.randint(-2, 2, (B, L, H, W), device=device).float() * step_y
    flows = torch.stack([shifts_x, shifts_y], dim=2)
    images = torch.randn(B, L, C, H, W, device=device)

    model = GridSamplePScan(mode='nearest', channels=C, use_decay=False, use_residual=False, chunk_size=2).to(device)

    with torch.no_grad():
        out_pscan = model(flows, images)
        out_ref = ref_flow_scan_additive(flows, images, mode='nearest')

    diff = (out_ref - out_pscan).abs().max().item()
    print(f"Max Difference: {diff:.2e}")

    if diff < 1e-5:
        print(">> Logic: PASSED")
    else:
        print(">> Logic: FAILED")

def check_chunk_consistency(device="cuda"):
    print("\n=== Chunk Consistency Check ===")
    
    B, L, C, H, W = 1, 16, 2, 32, 32
    flows = torch.randn(B, L, 2, H, W, device=device)
    images = torch.randn(B, L, C, H, W, device=device)

    model_small_chunk = GridSamplePScan(mode='bilinear', channels=C, use_decay=False, use_residual=False, chunk_size=2).to(device)
    model_large_chunk = GridSamplePScan(mode='bilinear', channels=C, use_decay=False, use_residual=False, chunk_size=L).to(device)

    with torch.no_grad():
        out_small = model_small_chunk(flows, images)
        out_large = model_large_chunk(flows, images)

    diff = (out_small - out_large).abs().max().item()
    print(f"Difference between chunk=2 and chunk={L}: {diff:.2e}")

    if diff < 1e-5:
        print(">> Chunk Check: PASSED")
    else:
        print(">> Chunk Check: FAILED")

def check_resolution_mismatch(device="cuda"):
    print("\n=== Resolution Decoupling Check ===")
    
    B, L, C, H, W = 2, 5, 3, 64, 64
    H_flow, W_flow = 32, 32 

    flows = torch.randn(B, L, 2, H_flow, W_flow, device=device)
    images = torch.randn(B, L, C, H, W, device=device)

    model = GridSamplePScan(mode='bilinear', channels=C, use_decay=False, use_residual=False, chunk_size=4).to(device)

    try:
        out = model(flows, images)
        print(f"Input Flow: {(H_flow, W_flow)}, Input Image: {(H, W)}")
        print(f"Output Shape: {out.shape}")
        if out.shape == (B, L, C, H, W):
            print(">> Resolution Check: PASSED")
        else:
            print(">> Resolution Check: FAILED (Wrong Shape)")
    except Exception as e:
        print(f">> Resolution Check: FAILED (Error: {e})")

def check_full_module_grad(device="cuda"):
    print("\n=== Full Module Gradient Check (With Decay & Residual) ===")

    B, L, C, H, W = 2, 4, 4, 32, 32
    flows = torch.randn(B, L, 2, H, W, device=device, requires_grad=True)
    images = torch.randn(B, L, C, H, W, device=device, requires_grad=True)

    model = GridSamplePScan(mode='bilinear', channels=C, use_decay=True, use_residual=True, chunk_size=2).to(device)
    
    out = model(flows, images)
    loss = out.sum()
    loss.backward()

    print(f"Flow Grad Norm: {flows.grad.norm().item():.2e}")
    print(f"Image Grad Norm: {images.grad.norm().item():.2e}")
    print(f"Decay Param Grad: {model.decay_log.grad is not None}")
    
    has_res_grad = False
    for name, param in model.res_conv.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            has_res_grad = True
            break
    print(f"Residual Network Grad: {has_res_grad}")

    if flows.grad is not None and images.grad is not None and has_res_grad:
         print(">> Gradient Check: PASSED")
    else:
         print(">> Gradient Check: FAILED")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_strict_logic(device)
    check_chunk_consistency(device)
    check_resolution_mismatch(device)
    check_full_module_grad(device)

