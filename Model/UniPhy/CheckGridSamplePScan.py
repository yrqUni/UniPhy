import torch
import torch.nn.functional as F
from GridSamplePScan import pscan_flow, flow_composition_residual

def ref_flow_scan(flows, images):
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
            img_t
        )
        out_images.append(curr_img_acc)
        
    return torch.stack(out_images, dim=1)

def check_approximation_error(device="cuda"):
    print(f"--- Check Approximation Error ---")
    
    torch.manual_seed(42)
    B, L, C, H, W = 2, 8, 4, 32, 32
    
    flows = torch.randn(B, L, 2, H, W, device=device) * 0.05
    images = torch.randn(B, L, C, H, W, device=device)
    
    with torch.no_grad():
        out_ref = ref_flow_scan(flows, images)
        out_pscan = pscan_flow(flows, images)

    diff = (out_ref - out_pscan).abs().mean().item()
    print(f"Mean Difference: {diff:.4f}")
    
    flows.requires_grad_(True)
    loss = pscan_flow(flows, images).sum()
    loss.backward()
    
    print(f"Backward Flow Grad Norm: {flows.grad.norm().item():.2e}")
    print(">> Check: DONE")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_approximation_error(device)

