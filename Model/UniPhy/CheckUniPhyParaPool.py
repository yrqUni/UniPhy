import torch
import torch.nn as nn
from UniPhyParaPool import UniPhyParaPool

def report(name, val, threshold=1e-5):
    passed = val < threshold
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"[{name}] Rel Error: {val:.2e} -> {color}{status}{reset}")
    if not passed:
        raise ValueError(f"{name} Failed")

def check_conservation():
    print("\n--- Checking UniPhyParaPool Conservation ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 4, 64, 32, 32
    model = UniPhyParaPool(dim=C).to(device)
    model.eval()
    
    x = torch.randn(B, C, H, W, dtype=torch.cfloat, device=device)
    
    with torch.no_grad():
        y = model(x)
        
    energy_in = torch.linalg.norm(x.flatten())
    energy_out = torch.linalg.norm(y.flatten())
    
    diff = abs(energy_in - energy_out)
    rel_error = diff / (energy_in + 1e-8)
    
    print(f"Energy In:  {energy_in:.6f}")
    print(f"Energy Out: {energy_out:.6f}")
    report("Norm Conservation", rel_error)

def check_gradient_flow():
    print("\n--- Checking Gradient Flow ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B, C, H, W = 2, 32, 16, 16
    model = UniPhyParaPool(dim=C).to(device)
    model.train()
    
    x = torch.randn(B, C, H, W, dtype=torch.cfloat, device=device, requires_grad=True)
    
    y = model(x)
    loss = y.abs().sum()
    loss.backward()
    
    has_grad = model.potential_net.net[0].weight.grad is not None
    grad_norm = model.potential_net.net[0].weight.grad.norm().item()
    
    print(f"Gradient exists: {has_grad}")
    print(f"Gradient norm:   {grad_norm:.6f}")
    
    if not has_grad or grad_norm == 0:
        raise ValueError("Gradient flow failed")
    else:
        print("\033[92mGradient Flow PASS\033[0m")

if __name__ == "__main__":
    torch.manual_seed(42)
    try:
        check_conservation()
        check_gradient_flow()
        print("\nAll checks passed.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")

