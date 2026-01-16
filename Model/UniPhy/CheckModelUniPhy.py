import torch
import torch.nn as nn
from ModelUniPhy import UniPhyModel

def report(name, val, threshold=1e-3, inverse=False):
    if inverse:
        passed = val > threshold
        status = "PASS" if passed else "FAIL (Too Small)"
    else:
        passed = val < threshold
        status = "PASS" if passed else "FAIL (Too Large)"
        
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"[{name}] Val: {val:.4e} -> {color}{status}{reset}")
    if not passed:
        raise ValueError(f"{name} Failed")

def check_model_structure():
    print("\n--- 1. Structure & Shape Check ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = 64, 128
    model = UniPhyModel(input_shape=(H, W), in_channels=4, dim=32, num_layers=2).to(device)
    
    B, T, C = 2, 5, 4
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device)
    
    z_pred, states = model(x, dt)
    
    print(f"Input: {x.shape}")
    print(f"Latent Output: {z_pred.shape}")
    
    expected_H = H // 4
    expected_W = W // 4
    if z_pred.shape[-2:] != (expected_H, expected_W):
        raise ValueError(f"Latent shape mismatch. Got {z_pred.shape[-2:]}, expected ({expected_H}, {expected_W})")
    
    print("\033[92mStructure Check PASS\033[0m")

def check_latent_conservation():
    print("\n--- 2. Latent Energy Conservation (Free Evolution) ---")
    print("Testing unitary evolution with Zero Input (System Isolation)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UniPhyModel(input_shape=(32, 32), in_channels=4, dim=64, num_layers=4, conserve_energy=True).to(device)
    model.eval()
    
    B, T, C, H, W = 1, 50, 4, 32, 32
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device) * 0.1
    
    with torch.no_grad():
        x_init = x[:, 0].unsqueeze(1)
        z_out, initial_states = model(x_init, dt[:, 0:1])
        
        dummy_x = torch.zeros_like(z_out[:, 0]).permute(0, 2, 3, 1) 
        
        h_energies = []
        
        curr_h = initial_states[0] 
        
        for t in range(T):
            next_states = []
            
            _, h_next_all = model.blocks[0].step_serial(dummy_x, dt[:, t], curr_h)
            
            energy = torch.norm(h_next_all)
            h_energies.append(energy)
            
            curr_h = h_next_all
            
    h_energies = torch.stack(h_energies)
    start_energy = h_energies[0]
    end_energy = h_energies[-1]
    
    max_dev = (h_energies - start_energy).abs().max() / (start_energy + 1e-6)
    
    print(f"Hidden State Energy (Start): {start_energy:.4f}")
    print(f"Hidden State Energy (End):   {end_energy:.4f}")
    
    report("Latent Energy Stability", max_dev, threshold=0.01)

def check_inference_loop():
    print("\n--- 3. Full Inference Loop (Encoder-Prop-Decoder) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    real_H, real_W = 64, 64
    model = UniPhyModel(input_shape=(real_H, real_W), in_channels=2, dim=32, num_layers=2).to(device)
    
    B, T_ctx, T_fut = 1, 5, 3
    C = 2
    
    ctx_x = torch.randn(B, T_ctx, C, real_H, real_W, device=device)
    ctx_dt = torch.rand(B, T_ctx, device=device)
    fut_dt = 0.1
    
    preds = model.inference(ctx_x, ctx_dt, T_fut, fut_dt, diffusion_steps=2)
    
    print(f"Prediction Shape: {preds.shape}")
    
    expected_shape = (B, T_fut, C, real_H, real_W)
    if preds.shape != expected_shape:
         raise ValueError(f"Inference output mismatch. Got {preds.shape}, expected {expected_shape}")
         
    pred_energy = torch.norm(preds)
    report("Prediction Signal Strength", pred_energy, threshold=1.0, inverse=True)

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    
    try:
        check_model_structure()
        check_latent_conservation()
        check_inference_loop()
        print("\nAll End-to-End Checks Passed.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")

