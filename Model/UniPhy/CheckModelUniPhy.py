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
    # 随机初始化
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device) * 0.1
    
    with torch.no_grad():
        # 1. Warmup: 给一个初始脉冲，建立 Hidden State
        x_init = x[:, 0].unsqueeze(1)
        z_out, initial_states = model(x_init, dt[:, 0:1])
        
        # 2. Free Evolution: 后续输入全为 0，观察 Hidden State 是否守恒
        # 注意：我们检查的是 Hidden State (h) 的能量，这才是哈密顿量承载的地方
        dummy_x = torch.zeros_like(z_out[:, 0]).permute(0, 2, 3, 1) # [B, H, W, D]
        
        h_energies = []
        
        # 追踪第一层的 Hidden State (最直接反映 Propagator 行为)
        curr_h = initial_states[0] 
        
        for t in range(T):
            next_states = []
            
            # 我们不仅跑 block，我们直接观测 block 内部 propagator 的 state 演化
            # 为了严谨，我们还是跑完整个 block，但输入是 0
            
            # 使用 step_serial, 输入为 0
            _, h_next_all = model.blocks[0].step_serial(dummy_x, dt[:, t], curr_h)
            
            # 记录能量
            energy = torch.norm(h_next_all)
            h_energies.append(energy)
            
            curr_h = h_next_all
            
    h_energies = torch.stack(h_energies)
    start_energy = h_energies[0]
    end_energy = h_energies[-1]
    
    # 允许微小的数值误差
    max_dev = (h_energies - start_energy).abs().max() / (start_energy + 1e-6)
    
    print(f"Hidden State Energy (Start): {start_energy:.4f}")
    print(f"Hidden State Energy (End):   {end_energy:.4f}")
    
    #  - 仅作示意，实际不生成
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

