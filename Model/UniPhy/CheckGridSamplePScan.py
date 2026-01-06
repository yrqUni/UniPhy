import torch
import torch.nn.functional as F
from GridSamplePScan import pscan_flow, flow_composition

def ref_flow_scan(grids, images):
    B, L, H, W, _ = grids.shape
    out_images = []
    
    # T=0
    curr_grid_acc = grids[:, 0]
    curr_img_acc = images[:, 0]
    out_images.append(curr_img_acc)
    
    # T=1..L-1
    for t in range(1, L):
        grid_t = grids[:, t]
        img_t = images[:, t]
        
        # Consistent with PScan: Accumulate = Prev (+) Curr
        curr_grid_acc, curr_img_acc = flow_composition(
            curr_grid_acc, 
            curr_img_acc, 
            grid_t, 
            img_t
        )
        out_images.append(curr_img_acc)
        
    return torch.stack(out_images, dim=1)

def check_consistency(device="cuda"):
    print(f"--- Running Consistency Check on {device} ---")
    torch.manual_seed(42)
    
    # Parameters
    B, L, C, H, W = 2, 8, 4, 32, 32
    
    # Inputs: Ensure grids are strictly within [-1, 1]
    # Small displacements are safer for stability tests
    grids = (torch.rand(B, L, H, W, 2, device=device) * 2 - 1) * 0.8
    images = torch.randn(B, L, C, H, W, device=device)
    
    # 1. Forward Check
    with torch.no_grad():
        out_ref = ref_flow_scan(grids, images)
        out_pscan = pscan_flow(grids, images)

    abs_diff = (out_ref - out_pscan).abs()
    print(f"Forward Max Diff: {abs_diff.max().item():.2e}")
    
    if abs_diff.max().item() < 1e-4:
        print(">> Forward Pass: PASSED")
    else:
        print(">> Forward Pass: FAILED")

    # 2. Backward Check
    grids_ref = grids.clone().requires_grad_(True)
    images_ref = images.clone().requires_grad_(True)
    grids_pscan = grids.clone().requires_grad_(True)
    images_pscan = images.clone().requires_grad_(True)
    
    loss_ref = ref_flow_scan(grids_ref, images_ref).sum()
    loss_ref.backward()
    
    loss_pscan = pscan_flow(grids_pscan, images_pscan).sum()
    loss_pscan.backward()
    
    grad_grid_diff = (grids_ref.grad - grids_pscan.grad).abs().max().item()
    grad_img_diff = (images_ref.grad - images_pscan.grad).abs().max().item()
    
    print(f"Backward Grid Grad Diff: {grad_grid_diff:.2e}")
    print(f"Backward Image Grad Diff: {grad_img_diff:.2e}")

    if grad_grid_diff < 1e-3 and grad_img_diff < 1e-3:
        print(">> Backward Pass: PASSED")
    else:
        print(">> Backward Pass: FAILED")

if __name__ == "__main__":
    if torch.cuda.is_available():
        check_consistency("cuda")
    else:
        check_consistency("cpu")

