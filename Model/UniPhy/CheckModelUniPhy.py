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
    print("\n--- 1. Structure & Shape Check (Diffusion vs Ensemble) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = 64, 128
    B, T, C = 2, 5, 4
    x = torch.randn(B, T, C, H, W, device=device)
    dt = torch.rand(B, T, device=device)
    for dtype in ['diffusion', 'ensemble']:
        print(f"Testing Decoder Type: {dtype}")
        model = UniPhyModel(input_shape=(H, W), in_channels=4, dim=32, num_layers=2, decoder_type=dtype).to(device)
        z_pred, states = model(x, dt)
        expected_H, expected_W = H // 4, W // 4
        if z_pred.shape[-2:] != (expected_H, expected_W):
            color = "\033[91m"
            reset = "\033[0m"
            raise ValueError(f"{dtype} Latent shape mismatch. {color}FAIL{reset}")
        color = "\033[92m"
        reset = "\033[0m"
        print(f"[{dtype}] Latent Output: {z_pred.shape} {color}PASS{reset}")

def check_latent_conservation():
    print("\n--- 2. Latent Energy Conservation (Free Evolution) ---")
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
            _, h_next_all = model.blocks[0].step_serial(dummy_x, dt[:, t], curr_h)
            energy = torch.norm(h_next_all)
            h_energies.append(energy)
            curr_h = h_next_all
    h_energies = torch.stack(h_energies)
    max_dev = (h_energies - h_energies[0]).abs().max() / (h_energies[0] + 1e-6)
    report("Latent Energy Stability", max_dev, threshold=0.01)

def check_inference_loop():
    print("\n--- 3. Full Inference Loop (Diff vs Ensemble) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_H, real_W = 64, 64
    B, T_ctx, T_fut, C = 1, 5, 3, 2
    ctx_x = torch.randn(B, T_ctx, C, real_H, real_W, device=device)
    ctx_dt = torch.rand(B, T_ctx, device=device)
    fut_dt = 0.1

    print("Testing Diffusion Inference...")
    m_diff = UniPhyModel(input_shape=(real_H, real_W), in_channels=C, dim=16, decoder_type='diffusion').to(device)
    p_diff = m_diff.inference(ctx_x, ctx_dt, T_fut, fut_dt, diffusion_steps=2)
    if p_diff.shape != (B, T_fut, C, real_H, real_W):
        raise ValueError(f"Diffusion shape mismatch: {p_diff.shape}")
    report("Diff Signal", torch.norm(p_diff), threshold=1.0, inverse=True)

    print("Testing Ensemble Inference...")
    m_ens = UniPhyModel(input_shape=(real_H, real_W), in_channels=C, dim=16, decoder_type='ensemble', ensemble_size=4).to(device)
    p_ens = m_ens.inference(ctx_x, ctx_dt, T_fut, fut_dt, num_ensemble=4)
    if p_ens.shape != (B, T_fut, 4, C, real_H, real_W):
        raise ValueError(f"Ensemble shape mismatch: {p_ens.shape}")
    report("Ensemble Signal", torch.norm(p_ens), threshold=1.0, inverse=True)

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    try:
        check_model_structure()
        check_latent_conservation()
        check_inference_loop()
        print("\nAll End-to-End Checks Passed.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nVerification Failed: {e}")

