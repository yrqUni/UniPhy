import torch
import math
import sys
import os
import time

try:
    from BarotropicVorticitySolver import BarotropicVorticitySolver
except ImportError:
    print("Error: Cannot import BarotropicVorticitySolver.py")
    sys.exit(1)

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"

def compute_invariants(zeta_phys, solver):
    zeta_hat = torch.fft.rfft2(zeta_phys)
    
    inv_lap = solver.inv_laplacian_k
    lap = solver.laplacian_k
    
    psi_hat = -zeta_hat * inv_lap
    
    energy_spec = 0.5 * torch.abs(psi_hat)**2 * torch.abs(lap)
    energy = torch.sum(energy_spec)
    
    enstrophy_spec = 0.5 * torch.abs(zeta_hat)**2
    enstrophy = torch.sum(enstrophy_spec)
    
    return energy.item(), enstrophy.item()

def run_check(device_name):
    H, W = 128, 128
    dt = 0.01
    steps = 100
    
    device = torch.device(device_name)
    print(f"{BOLD}[Checking Device]{RESET} {CYAN}{device}{RESET}")

    print(f"{BOLD}[Test 1: Inviscid Conservation (nu=0)]{RESET}")
    solver = BarotropicVorticitySolver(H, W, dt=dt, viscosity=0.0).to(device)
    
    x = torch.linspace(0, 2*math.pi, W, device=device)
    y = torch.linspace(0, 2*math.pi, H, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    
    zeta_0 = torch.sin(4*grid_x) * torch.sin(4*grid_y) + \
             0.5 * torch.cos(3*grid_x) * torch.sin(5*grid_y)
    zeta_0 = zeta_0.unsqueeze(0).unsqueeze(0) 

    E0, Z0 = compute_invariants(zeta_0, solver)
    print(f"  Initial Energy:    {E0:.6e}")
    print(f"  Initial Enstrophy: {Z0:.6e}")

    zeta_t = zeta_0
    start_t = time.time()
    with torch.no_grad():
        for i in range(steps):
            zeta_t = solver(zeta_t, steps=1)
    end_t = time.time()
    
    Et, Zt = compute_invariants(zeta_t, solver)
    
    err_E = abs((Et - E0) / E0)
    err_Z = abs((Zt - Z0) / Z0)
    
    fps = steps / (end_t - start_t)
    
    print(f"  Final Energy:      {Et:.6e}")
    print(f"  Final Enstrophy:   {Zt:.6e}")
    
    if err_E < 1e-4:
        c_e = GREEN
    elif err_E < 1e-2:
        c_e = YELLOW
    else:
        c_e = RED
        
    if err_Z < 1e-4:
        c_z = GREEN
    elif err_Z < 1e-2:
        c_z = YELLOW
    else:
        c_z = RED

    print(f"  Rel Error Energy:  {c_e}{err_E:.2e}{RESET}")
    print(f"  Rel Error Enstrophy: {c_z}{err_Z:.2e}{RESET}")
    print(f"  Speed: {GRAY}{fps:.1f} steps/sec{RESET}\n")

    print(f"{BOLD}[Test 2: Viscous Decay (nu=1e-3)]{RESET}")
    solver_visc = BarotropicVorticitySolver(H, W, dt=dt, viscosity=1e-3).to(device)
    
    E0_v, Z0_v = compute_invariants(zeta_0, solver_visc)
    
    zeta_tv = zeta_0
    with torch.no_grad():
        for i in range(steps):
            zeta_tv = solver_visc(zeta_tv, steps=1)
            
    Et_v, Zt_v = compute_invariants(zeta_tv, solver_visc)
    
    print(f"  Initial Enstrophy: {Z0_v:.6e}")
    print(f"  Final Enstrophy:   {Zt_v:.6e}")
    
    if Zt_v < Z0_v:
        print(f"  Decay Check:       {GREEN}PASS (Decreased){RESET}")
    else:
        print(f"  Decay Check:       {RED}FAIL (Not Decreased){RESET}")
    print("-" * 40)

def main():
    torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        run_check("cuda")
    else:
        run_check("cpu")

if __name__ == "__main__":
    main()

