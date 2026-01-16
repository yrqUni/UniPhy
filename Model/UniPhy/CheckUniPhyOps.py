import torch
import torch.nn as nn
from UniPhyOps import UniPhyLayer

def calculate_divergence(seq):
    B, T, C, H, W = seq.shape
    u = seq[:, :, 0, :, :]
    v = seq[:, :, 1, :, :]
    
    grad_u_x = torch.gradient(u, dim=-1)[0]
    grad_v_y = torch.gradient(v, dim=-2)[0]
    
    div = grad_u_x + grad_v_y
    return torch.abs(div).mean()

def calculate_energy(seq):
    return torch.sum(seq**2, dim=(2, 3, 4))

def run_verification():
    B, T, C, H, W = 2, 10, 4, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UniPhyLayer(emb_ch=C, input_shape=(H, W)).to(device)
    
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device) * 0.1
    
    try:
        out = model(x, dt)
        print(f"Forward Pass: SUCCESS")
        print(f"Output Shape: {out.shape}")
        assert out.shape == (B, T, C, H, W)
    except Exception as e:
        print(f"Forward Pass: FAILED | Error: {e}")
        return

    div_val = calculate_divergence(out)
    print(f"Mean Divergence: {div_val.item():.2e}")
    
    if div_val < 1e-5:
        print("Divergence Check: PASSED (Solenoidal field maintained)")
    else:
        print("Divergence Check: WARNING (High divergence detected)")

    energies = calculate_energy(out)
    energy_ratios = energies[:, -1] / energies[:, 0]
    
    print(f"Energy at t=0: {energies[0, 0].item():.2f}")
    print(f"Energy at t=T: {energies[0, -1].item():.2f}")
    
    is_stable = torch.all(energies < 1e5)
    if is_stable:
        print("Stability Check: PASSED (No numerical explosion)")
    else:
        print("Stability Check: FAILED (System exploded)")

    with torch.no_grad():
        nn.init.zeros_(model.gen_proj.weight)
        nn.init.zeros_(model.gen_proj.bias)
        
        out_pure = model(x, dt)
        energies_pure = calculate_energy(out_pure)
        
        drift = torch.abs(energies_pure[:, -1] - energies_pure[:, 0]).mean()
        print(f"Ideal Propagation Drift: {drift.item():.2e}")

if __name__ == "__main__":
    run_verification()

