import torch
import torch.nn.functional as F
from GridSamplePScan import pscan_flow, flow_composition_residual

def ref_flow_scan(flows, images, mode='bilinear'):
    B, L, _, H, W = flows.shape
    out_images = []
    
    curr_flow_acc = flows[:, 0]
    curr_img_acc = images[:, 0]
    out_images.append(curr_img_acc)
    
    for t in range(1, L):
        flow_t = flows[:, t]
        img_t = images[:, t]
        
        curr_flow_acc, curr_img_acc = flow_composition_residual(
            curr_flow_acc, 
            curr_img_acc, 
            flow_t, 
            img_t,
            mode=mode
        )
        out_images.append(curr_img_acc)
        
    return torch.stack(out_images, dim=1)

def check_strict_logic(device="cuda"):
    print(f"\n=== Test 1: Strict Logic Check (Integer Flow + Nearest) ===")
    torch.manual_seed(42)
    B, L, C, H, W = 1, 8, 1, 32, 32
    
    step_y = 2.0 / H
    step_x = 2.0 / W
    
    shifts_x = torch.randint(-10, 10, (B, L, H, W), device=device).float() * step_x
    shifts_y = torch.randint(-5, 5, (B, L, H, W), device=device).float() * step_y
    
    flows = torch.stack([shifts_x, shifts_y], dim=2)
    images = torch.randn(B, L, C, H, W, device=device)

    with torch.no_grad():
        out_ref = ref_flow_scan(flows, images, mode='nearest')
        out_pscan = pscan_flow(flows, images, mode='nearest')

    diff = (out_ref - out_pscan).abs().max().item()
    print(f"Max Difference: {diff:.2e}")
    if diff < 1e-5:
        print(">> Logic Check: PASSED")
    else:
        print(">> Logic Check: FAILED")

def check_approximation_error(device="cuda"):
    print(f"\n=== Test 2: Approximation Error (Float Flow + Bilinear) ===")
    torch.manual_seed(42)
    B, L, C, H, W = 2, 8, 4, 32, 32
    
    flows = torch.randn(B, L, 2, H, W, device=device) * 0.05
    images = torch.randn(B, L, C, H, W, device=device)
    
    with torch.no_grad():
        out_ref = ref_flow_scan(flows, images, mode='bilinear')
        out_pscan = pscan_flow(flows, images, mode='bilinear')

    diff = (out_ref - out_pscan).abs().mean().item()
    print(f"Mean Difference: {diff:.4f}")
    
    flows.requires_grad_(True)
    loss = pscan_flow(flows, images, mode='bilinear').sum()
    loss.backward()
    
    print(f"Backward Flow Grad Norm: {flows.grad.norm().item():.2e}")
    print(">> Check: DONE")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_strict_logic(device)
    check_approximation_error(device)

