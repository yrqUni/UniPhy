import torch
import torch.nn as nn
from UniPhyOps import UniPhyLayer
import traceback

def main():
    print("=== [Advanced Check] UniPhyOps Numerical & Physics Integrity ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    B, C, H, W = 2, 32, 64, 64
    emb_ch = C
    rank = 16
    dt_val = 0.1
    
    try:
        model = UniPhyLayer(emb_ch=emb_ch, input_shape=(H, W), rank=rank).to(device)
        print(f"[Pass] Model Initialization")
    except Exception as e:
        print(f"[Fail] Model Initialization: {e}")
        traceback.print_exc()
        return

    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    dt = torch.ones(B, device=device) * dt_val
    h_prev = None

    print("\n--- 1. Forward & Shape Check ---")
    try:
        out_1, h_1 = model(x, h_prev, dt)
        assert out_1.shape == (B, C, H, W), f"Output shape error: {out_1.shape}"
        assert not torch.isnan(out_1).any(), "NaN detected in output"
        print(f"[Pass] Forward Step 1: Shape {tuple(out_1.shape)}")
        
        out_2, h_2 = model(x * 0.5, h_1, dt)
        assert not torch.allclose(h_1, h_2), "Hidden state not evolving"
        assert h_2.shape == (B, C, H, W), "Hidden state shape error"
        print(f"[Pass] Forward Step 2 (Recursive): Hidden state evolved")
    except Exception as e:
        print(f"[Fail] Forward Check: {e}")
        traceback.print_exc()
        return

    print("\n--- 2. Divergence Elimination Check (Helmholtz) ---")
    try:
        def compute_div(v_field):
            u = v_field[:, 0:1]
            v = v_field[:, 1:2]
            kx = model.projection_op.kx
            ky = model.projection_op.ky
            u_f = torch.fft.fftn(u, dim=(-2, -1))
            v_f = torch.fft.fftn(v, dim=(-2, -1))
            div_f = 1j * kx * u_f + 1j * ky * v_f
            return torch.fft.ifftn(div_f, dim=(-2, -1)).real.norm()

        div_before = compute_div(out_1)
        out_projected = model.projection_op(out_1)
        div_after = compute_div(out_projected)
        
        print(f"Divergence Norm: {div_before.item():.6f} -> {div_after.item():.6f}")
        if div_after < div_before * 0.1 or div_after < 1e-4:
            print("[Pass] Helmholtz Projection effectively reduced divergence")
        else:
            print("[Warning] Divergence reduction not significant")
    except Exception as e:
        print(f"[Error] During Divergence Check: {e}")

    print("\n--- 3. Gradient Flow & Orthogonality Check ---")
    try:
        loss = out_2.pow(2).mean()
        loss.backward()
        
        grad_map = {name: p.grad is not None for name, p in model.named_parameters()}
        
        checks = {
            "Transport (Advection)": "transport_op.net",
            "Interaction (Clifford)": "interaction_op.0",
            "Dispersion (Spectral)": "dispersion_op.estimator",
            "Mixing (StreamFunc)": "stream_mixing_op.psi_net"
        }
        
        all_passed = True
        for logic_name, param_prefix in checks.items():
            found = any(name.startswith(param_prefix) and has_grad for name, has_grad in grad_map.items())
            status = "[OK]" if found else "[MISSING]"
            print(f"  {status} {logic_name} gradient path")
            if not found: all_passed = False
            
        if all_passed:
            print("[Pass] All physical operators are connected to the computational graph")
        else:
            print("[Fail] Some physical paths are disconnected (Gradients = None)")

        print(f"\nGradient Magnitude Samples:")
        t_grad = model.transport_op.net[0].weight.grad.abs().mean().item()
        i_grad = model.interaction_op[2].weight.grad.abs().mean().item()
        print(f"  Transport Grad Avg: {t_grad:.8e}")
        print(f"  Interaction Grad Avg: {i_grad:.8e}")

    except Exception as e:
        print(f"[Fail] Backward Check: {e}")
        traceback.print_exc()

    print("\n=== All Operations Checked ===")

if __name__ == "__main__":
    main()

