import torch
import torch.nn as nn
import time
import sys
import numpy as np

try:
    from SymplecticManifoldOperator import SymplecticManifoldNet
except ImportError:
    print("Error: Could not import SymplecticManifoldOperator.")
    sys.exit(1)

def check_symplectic_manifold():
    print("="*60)
    print("Test: Symplectic Manifold Neural Operator (Strang Splitting)")
    print("="*60)

    batch_size = 2
    in_ch = 4
    out_ch = 4
    h, w = 64, 128
    hidden_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SymplecticManifoldNet(in_ch=in_ch, out_ch=out_ch, hidden_dim=hidden_dim, h=h, w=w).to(device)
    print("Model built: Quantum Dynamics + Diffeomorphic Advection")

    x = torch.randn(batch_size, in_ch, h, w).to(device)
    dt = 1.0

    print("-" * 30)
    print("1. Forward Pass (Strang Splitting Step)...")
    start_time = time.time()
    y = model(x, dt=dt)
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f}s")
    print(f"Output shape: {y.shape}")

    if x.shape != y.shape:
        print("FAIL: Shape mismatch.")
        return
    else:
        print("PASS: Dimension check.")

    print("-" * 30)
    print("2. Gradient Flow Check...")
    loss = torch.mean((y - x) ** 2)
    model.zero_grad()
    loss.backward()

    hamiltonian_grad = model.kinetic_op.hamiltonian_real.grad is not None
    flow_grad = False
    for p in model.advection_op.flow_net.parameters():
        if p.grad is not None:
            flow_grad = True
            break
            
    print(f"Quantum Hamiltonian Gradient: {'YES' if hamiltonian_grad else 'NO'}")
    print(f"Diffeomorphic Flow Gradient: {'YES' if flow_grad else 'NO'}")

    if hamiltonian_grad and flow_grad:
        print("PASS: All physics components are learnable.")
    else:
        print("FAIL: Gradient flow broken.")

    print("-" * 30)
    print("3. Time-Reversibility Check (Geometric Quality)...")
    with torch.no_grad():
        z0 = torch.complex(model.encoder_real(x), model.encoder_imag(x))
        
        z_half = model.kinetic_op(z0, dt/2)
        z_adv = model.advection_op(z_half, dt)
        z_final = model.kinetic_op(z_adv, dt/2)
        
        z_rev_half = model.kinetic_op(z_final, -dt/2)
        z_rev_adv = model.advection_op(z_rev_half, -dt) 
        z_rev_final = model.kinetic_op(z_rev_adv, -dt/2)
        
        recon_error = torch.mean(torch.abs(z_rev_final - z0)).item()
        print(f"Reversibility Error (Structure Check): {recon_error:.6f}")
        
        if recon_error < 1.0: 
            print("PASS: Structure supports approximate reversibility.")
        else:
            print("WARNING: High irreversibility.")

    print("="*60)
    print("Test Complete.")

if __name__ == "__main__":
    check_symplectic_manifold()

