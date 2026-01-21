import torch
import numpy as np
import matplotlib.pyplot as plt
from models_physics import DiscreteBaseline, UniPhyPhysicsAdapter

def analyze():
    device = torch.device("cuda")
    N = 64
    model_base = DiscreteBaseline(N=N).to(device)
    model_uni = UniPhyPhysicsAdapter(N=N).to(device)
    model_base.load_state_dict(torch.load("base_wave.pth"))
    model_uni.load_state_dict(torch.load("uniphy_wave.pth"))
    
    x = np.linspace(0, 1, N, endpoint=False)
    u0 = np.exp(-100 * (x - 0.3)**2)
    u0_t = torch.tensor(u0, dtype=torch.float32).unsqueeze(0).to(device)
    
    dt_test = 0.5
    with torch.no_grad():
        pred_uni = model_uni(u0_t, torch.tensor([[dt_test]], device=device)).cpu().numpy()[0]
        u1_base = model_base(u0_t).cpu().numpy()[0]
        pred_interp = (u0 + u1_base) / 2.0 
        
    gt = np.exp(-100 * (np.mod(x - 0.3 - 0.1 + 0.5, 1.0) - 0.5)**2)

    plt.figure(figsize=(12, 6))
    plt.plot(x, u0, 'k--', label='Initial (t=0)')
    plt.plot(x, gt, 'k-', label='Ground Truth (t=0.5)', lw=3)
    plt.plot(x, pred_interp, 'r:', label='Linear Interpolation (Ghosting/Two Peaks)')
    plt.plot(x, pred_uni, 'b-', label='UniPhy (Continuous Phase Shift)', lw=2)
    plt.legend()
    plt.title("Zero-shot Temporal Generalization: Phase Shift vs. Ghosting")
    plt.savefig("exp3_physics_comparison.png")
    
    print(f"Energy (L2 Norm) at t=0.5:")
    print(f"GT: {np.linalg.norm(gt):.4f}")
    print(f"UniPhy: {np.linalg.norm(pred_uni):.4f}")
    print(f"Interpolation: {np.linalg.norm(pred_interp):.4f}")

if __name__ == "__main__":
    analyze()

