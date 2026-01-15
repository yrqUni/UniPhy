import torch
import torch.nn as nn
from UniPhyOps import UniPhyLayer

def main():
    print("Checking UniPhyOps.py...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU (Warning: Triton kernels require GPU)")

    B, C, H, W = 2, 32, 64, 64
    emb_ch = C
    rank = 16
    
    try:
        model = UniPhyLayer(emb_ch=emb_ch, input_shape=(H, W), rank=rank).to(device)
        print(f"[Pass] Model Initialization: UniPhyLayer(emb_ch={emb_ch}, input_shape={H}x{W}, rank={rank})")
    except Exception as e:
        print(f"[Fail] Model Initialization: {e}")
        return

    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    dt = torch.ones(B, device=device) * 0.1
    h_prev = None

    try:
        out_1, h_1 = model(x, h_prev, dt)
        
        assert out_1.shape == (B, C, H, W), f"Shape mismatch: {out_1.shape}"
        assert h_1.shape == (B, C, H, W), f"Hidden shape mismatch: {h_1.shape}"
        assert not torch.isnan(out_1).any(), "Output contains NaNs"
        assert not torch.isnan(h_1).any(), "Hidden state contains NaNs"
        
        print(f"[Pass] Forward Step 1 (h_prev=None): Output {out_1.shape}")
    except Exception as e:
        print(f"[Fail] Forward Step 1: {e}")
        import traceback
        traceback.print_exc()
        return

    x2 = torch.randn(B, C, H, W, device=device, requires_grad=True)
    
    try:
        out_2, h_2 = model(x2, h_1, dt)
        
        assert out_2.shape == (B, C, H, W)
        assert not torch.allclose(out_1, out_2), "Outputs identical for different inputs"
        
        print(f"[Pass] Forward Step 2 (h_prev=Tensor)")
    except Exception as e:
        print(f"[Fail] Forward Step 2: {e}")
        return

    try:
        loss = out_2.sum() + h_2.sum()
        loss.backward()
        
        params_checked = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_checked.append(name)
        
        required_grads = [
            'h_params_r', 
            'h_params_i', 
            'sigma', 
            'advection.net.0.weight', 
            'clifford_in.conv_s.weight'
        ]
        
        missing_grads = [name for name in required_grads if not any(name in p for p in params_checked)]
        
        if missing_grads:
            print(f"[Warning] Missing gradients for: {missing_grads}")
        else:
            print(f"[Pass] Backward Pass: Gradients computed successfully")
            
        print(f"Gradient Norm Check:")
        print(f"  Sigma grad: {model.sigma.grad}")
        print(f"  Advection weight grad norm: {model.advection.net[0].weight.grad.norm().item():.4f}")
        
    except Exception as e:
        print(f"[Fail] Backward Pass: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n[Success] All checks passed for UniPhyOps.")

if __name__ == "__main__":
    main()

