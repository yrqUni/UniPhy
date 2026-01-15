import torch
import torch.nn as nn
import time
import sys

try:
    from GrandUnifiedModel import GeoSymSDE
except ImportError:
    print("Error: Could not import GeoSymSDE.")
    sys.exit(1)

def check_grand_unified():
    print("="*70)
    print("Test: GeoSym-SDE (Clifford + Hamiltonian + StreamNet + SDE + Triton)")
    print("="*70)

    batch_size = 2
    in_ch = 4
    out_ch = 4
    h, w = 32, 64
    hidden_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GeoSymSDE(in_ch=in_ch, out_ch=out_ch, hidden_dim=hidden_dim, h=h, w=w).to(device)
    print("Model Architecture: Grand Unified Physics-AI built.")

    x = torch.randn(batch_size, in_ch, h, w).to(device)
    dt = 1.0

    print("-" * 30)
    print("1. Forward Pass & Stochasticity Check...")
    y1 = model(x, dt=dt)
    y2 = model(x, dt=dt)
    
    print(f"Output shape: {y1.shape}")
    
    diff = torch.mean(torch.abs(y1 - y2)).item()
    print(f"Ensemble Divergence (Noise Effect): {diff:.6f}")
    
    if diff > 1e-6:
        print("PASS: Model successfully generates probabilistic ensembles.")
    else:
        print("FAIL: Model is deterministic (SDE inactive).")

    print("-" * 30)
    print("2. Physical Constraint Check (Incompressibility)...")
    with torch.no_grad():
        geo_feat = model.clifford_encoder(x)
        z_real = model.to_complex_real(geo_feat)
        psi = model.stream_op.psi_net(z_real)
        
        psi_pad = torch.nn.functional.pad(psi, (1,1,1,1))
        u = (psi_pad[:, :, 2:, 1:-1] - psi_pad[:, :, :-2, 1:-1]) / 2.0
        v = -(psi_pad[:, :, 1:-1, 2:] - psi_pad[:, :, 1:-1, :-2]) / 2.0
        
        u_pad = torch.nn.functional.pad(u, (1,1,1,1))
        v_pad = torch.nn.functional.pad(v, (1,1,1,1))
        du_dx = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / 2.0
        dv_dy = (v_pad[:, 2:, 1:-1] - v_pad[:, :-2, 1:-1]) / 2.0
        
        div = torch.mean(torch.abs(du_dx + dv_dy)).item()
        print(f"Internal Flow Divergence: {div:.8f}")
        
        if div < 1e-5:
            print("PASS: Hard physical constraint (div=0) active.")
        else:
            print("FAIL: Physical constraint broken.")

    print("-" * 30)
    print("3. Gradient Flow Check...")
    loss = torch.mean(y1 ** 2)
    model.zero_grad()
    loss.backward()
    
    clifford_grad = model.clifford_encoder[1].weight_b.grad is not None
    hamiltonian_grad = model.stochastic_op.hamiltonian_real.grad is not None
    stream_grad = model.stream_op.psi_net[0].weight.grad is not None
    noise_grad = model.stochastic_op.noise_scale.grad is not None
    
    print(f"Clifford Algebra Grads: {'YES' if clifford_grad else 'NO'}")
    print(f"Hamiltonian Spec Grads: {'YES' if hamiltonian_grad else 'NO'}")
    print(f"Stream Function Grads:  {'YES' if stream_grad else 'NO'}")
    print(f"SDE Noise Param Grads:  {'YES' if noise_grad else 'NO'}")

    if clifford_grad and hamiltonian_grad and stream_grad and noise_grad:
        print("PASS: All systems (Algebra, Geometry, Probability) learning.")
    else:
        print("FAIL: Gradient flow broken.")

    print("="*70)
    print("Test Complete.")

if __name__ == "__main__":
    check_grand_unified()

