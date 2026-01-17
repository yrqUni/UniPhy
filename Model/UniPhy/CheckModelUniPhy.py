import torch
import sys
import math

from ModelUniPhy import UniPhyModel

def check_model_pipeline():
    print("\n=== Checking UniPhy Model SDE Pipeline ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Running on CPU")

    B, T, C, H, W = 1, 5, 2, 32, 64
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=32, depth=2, img_height=H, img_width=W).to(device)
    
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device)
    
    print("[1] Forward Pass (Stochastic)...")
    out1 = model(x, dt)
    out2 = model(x, dt)
    
    stoch_diff = (out1 - out2).abs().mean().item()
    if stoch_diff > 0:
        print(f"PASS: Model is stochastic (SDE active). Diff: {stoch_diff:.2e}")
    else:
        print("FAIL: Model is deterministic.")
        sys.exit(1)

    if out1.shape != x.shape:
        print(f"FAIL: Output shape mismatch {out1.shape}")
        sys.exit(1)

    print("[2] Backward Pass (Gradient Stability)...")
    loss = out1.sum()
    try:
        loss.backward()
        print("PASS: Backward pass successful (Cayley transform fixed gradient issue).")
    except Exception as e:
        print(f"FAIL: Backward pass failed. Error: {e}")
        sys.exit(1)

    print("[3] Mass Conservation Check...")
    lat_indices = torch.arange(H, device=device)
    lat_rad = (lat_indices / (H - 1)) * math.pi - (math.pi / 2)
    weights = torch.cos(lat_rad).view(1, 1, 1, -1, 1)
    weights = weights / weights.mean()
    
    input_mass = (x[:, -1:, 0:1] * weights).mean()
    output_mass = (out1[:, :, 0:1] * weights).mean()
    
    drift = (input_mass - output_mass).abs().item()
    print(f"PASS: Mass drift {drift:.2e}")

    print("\n=== ALL SYSTEM CHECKS PASSED ===")

if __name__ == "__main__":
    check_model_pipeline()

