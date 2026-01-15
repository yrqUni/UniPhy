import torch
import torch.nn as nn
import time
import sys
import numpy as np

try:
    from ConservativeSymplecticOperator import ConservativeSymplecticNet
except ImportError:
    print("Error: Could not import ConservativeSymplecticOperator.")
    sys.exit(1)

def check_conservative_symplectic():
    print("="*60)
    print("Test: Conservative Symplectic Net (Stream Function Constraints)")
    print("="*60)

    batch_size = 2
    in_ch = 4
    out_ch = 4
    h, w = 64, 128
    hidden_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ConservativeSymplecticNet(in_ch=in_ch, out_ch=out_ch, hidden_dim=hidden_dim, h=h, w=w).to(device)
    print("Model built: Quantum Dynamics + Stream Function Advection")

    x = torch.randn(batch_size, in_ch, h, w).to(device)
    dt = 1.0

    print("-" * 30)
    print("1. Forward Pass...")
    start_time = time.time()
    y = model(x, dt=dt)
    end_time = time.time()
    
    print(f"Time: {end_time - start_time:.4f}s")
    print(f"Output shape: {y.shape}")
    
    if x.shape != y.shape:
        print("FAIL: Shape mismatch.")
        return

    print("-" * 30)
    print("2. Conservation Law Check: Incompressibility (Div == 0)")
    with torch.no_grad():
        z = torch.complex(model.encoder_real(x), model.encoder_imag(x))
        psi = model.advection_op.psi_net(z.real)
        velocity = model.advection_op.compute_curl(psi)
        
        u = velocity[:, 0]
        v = velocity[:, 1]
        
        u_pad = torch.nn.functional.pad(u, (1,1,1,1))
        v_pad = torch.nn.functional.pad(v, (1,1,1,1))
        
        du_dx = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / 2.0
        dv_dy = (v_pad[:, 2:, 1:-1] - v_pad[:, :-2, 1:-1]) / 2.0
        
        divergence = du_dx + dv_dy
        mean_div = torch.mean(torch.abs(divergence)).item()
        max_div = torch.max(torch.abs(divergence)).item()
        
        print(f"Mean Divergence: {mean_div:.8f}")
        print(f"Max Divergence:  {max_div:.8f}")
        
        if mean_div < 1e-6:
            print("PASS: Hard Constraint holds. Flow is incompressible.")
        else:
            print("WARNING: Divergence is non-zero (check discretization).")

    print("-" * 30)
    print("3. Gradient Flow Check...")
    loss = torch.mean((y - x) ** 2)
    model.zero_grad()
    loss.backward()

    hamiltonian_grad = model.kinetic_op.hamiltonian_real.grad is not None
    psi_grad = False
    for p in model.advection_op.psi_net.parameters():
        if p.grad is not None:
            psi_grad = True
            break
            
    print(f"Quantum Hamiltonian Gradient: {'YES' if hamiltonian_grad else 'NO'}")
    print(f"Stream Function Gradient: {'YES' if psi_grad else 'NO'}")

    if hamiltonian_grad and psi_grad:
        print("PASS: Gradients flow through physical constraints.")
    else:
        print("FAIL: Gradient flow broken.")

    print("="*60)
    print("Test Complete.")

if __name__ == "__main__":
    check_conservative_symplectic()

