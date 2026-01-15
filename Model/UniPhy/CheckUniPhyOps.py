import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from UniPhyOps import UniPhyFluidScan

def compute_metrics(state, ky, kx):
    u_v = state[0, 0, :2, :, :]
    energy = 0.5 * torch.mean(u_v**2)
    xf = torch.fft.rfft2(u_v, dim=(-2, -1))
    u_f = xf[0]
    v_f = xf[1]
    div_f = 1j * ky * u_f + 1j * kx * v_f
    max_div = torch.abs(div_f).max()
    return energy.item(), max_div.item()

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 1, 4, 128, 128
    steps = 500
    dt = torch.tensor([0.02], device=device)
    
    model = UniPhyFluidScan(C, H, W).to(device)
    
    x = torch.linspace(0, 2*np.pi, W, device=device)
    y = torch.linspace(0, 2*np.pi, H, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    u = torch.sin(grid_x) * torch.cos(grid_y)
    v = -torch.cos(grid_x) * torch.sin(grid_y)
    t_field = torch.exp(-((grid_x-np.pi)**2 + (grid_y-np.pi)**2))
    q_field = torch.zeros_like(u)
    
    state = torch.stack([u, v, t_field, q_field], dim=0).unsqueeze(0).unsqueeze(1)
    
    ky_stat = torch.fft.fftfreq(H, device=device).view(H, 1)
    kx_stat = torch.fft.rfftfreq(W, device=device).view(1, W // 2 + 1)
    
    e_init, d_init = compute_metrics(state, ky_stat, kx_stat)
    print(f"Initial State | Energy: {e_init:.6f} | Div: {d_init:.2e}")
    
    energy_history = [e_init]
    div_history = [d_init]
    
    print("Starting simulation...")
    for i in range(steps):
        with torch.no_grad():
            state = model(state, dt)
            e, d = compute_metrics(state, ky_stat, kx_stat)
            energy_history.append(e)
            div_history.append(d)
            
            if (i + 1) % 50 == 0:
                print(f"Step {i+1:03d} | Energy: {e:.6f} | Div: {d:.2e} | Mag: {torch.norm(state).item():.4e}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(energy_history)
    plt.title("Energy Conservation")
    plt.xlabel("Step")
    
    plt.subplot(1, 3, 2)
    plt.plot(div_history)
    plt.yscale('log')
    plt.title("Divergence Stability")
    plt.xlabel("Step")
    
    plt.subplot(1, 3, 3)
    final_u = state[0, 0, 0].cpu().numpy()
    final_v = state[0, 0, 1].cpu().numpy()
    Y, X = np.mgrid[0:H, 0:W]
    plt.streamplot(X, Y, final_u, final_v, density=1.2)
    plt.title(f"Flow Field (Step {steps})")
    
    plt.tight_layout()
    plt.savefig("stability_report.png")

if __name__ == "__main__":
    run_benchmark()

