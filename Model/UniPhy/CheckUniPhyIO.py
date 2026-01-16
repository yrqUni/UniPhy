import torch
import torch.nn as nn
from UniPhyIO import UniPhyEncoder, UniPhyDiffusionDecoder, UniPhyEnsembleDecoder

def report(name, val, threshold=1e-5):
    passed = val < threshold
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"[{name}] Error/Metric: {val:.2e} -> {color}{status}{reset}")
    if not passed:
        raise ValueError(f"{name} Failed")

def check_geometric_conservation():
    print("\n--- Checking Geometric Conservation (Padding/Unpadding) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 30, 721, 1440
    patch_size = 4
    latent_dim = 64
    encoder = UniPhyEncoder(in_ch=C, embed_dim=latent_dim, patch_size=patch_size).to(device)
    diff_decoder = UniPhyDiffusionDecoder(out_ch=C, latent_dim=latent_dim, patch_size=patch_size).to(device)
    ens_decoder = UniPhyEnsembleDecoder(out_ch=C, latent_dim=latent_dim, patch_size=patch_size).to(device)
    x = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device).float()
    with torch.no_grad():
        z = encoder(x)
        x_diff = diff_decoder(z, x, t)
        x_ens = ens_decoder(z, x)
    shape_diff_diff = sum([abs(s1 - s2) for s1, s2 in zip(x_diff.shape, x.shape)])
    shape_diff_ens = sum([abs(s1 - s2) for s1, s2 in zip(x_ens.shape, x.shape)])
    report("Diff Decoder Shape", shape_diff_diff, threshold=0.1)
    report("Ensemble Decoder Shape", shape_diff_ens, threshold=0.1)

def check_ensemble_diversity():
    print("\n--- Checking Ensemble Diversity ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 1, 4, 32, 32
    ensemble_size = 5
    encoder = UniPhyEncoder(in_ch=C, embed_dim=16).to(device)
    decoder = UniPhyEnsembleDecoder(out_ch=C, latent_dim=16, ensemble_size=ensemble_size).to(device)
    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        z = encoder(x)
        ens_out = decoder.generate_ensemble(z, x, num_members=ensemble_size)
    if ens_out.shape != (B, ensemble_size, C, H, W):
        raise ValueError(f"Ensemble Output Shape Error: {ens_out.shape}")
    var = torch.var(ens_out, dim=1).mean().item()
    print(f"Ensemble Variance: {var:.6f}")
    report("Ensemble Diversity", 0.0 if var > 1e-10 else 1.0, threshold=0.1)

def check_gradient_flow():
    print("\n--- Checking Gradient Flow for Both Heads ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 1, 4, 32, 32
    encoder = UniPhyEncoder(in_ch=C, embed_dim=16).to(device)
    diff_decoder = UniPhyDiffusionDecoder(out_ch=C, latent_dim=16).to(device)
    ens_decoder = UniPhyEnsembleDecoder(out_ch=C, latent_dim=16).to(device)
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    z = encoder(x)
    pred_diff = diff_decoder(z, x, torch.ones(B, device=device))
    loss_diff = pred_diff.sum()
    loss_diff.backward(retain_graph=True)
    diff_grad_norm = torch.norm(x.grad).item()
    x.grad.zero_()
    pred_ens = ens_decoder(z, x)
    loss_ens = pred_ens.sum()
    loss_ens.backward()
    ens_grad_norm = torch.norm(x.grad).item()
    report("Diff Gradient Flow", 0.0 if diff_grad_norm > 0 else 1.0, threshold=0.1)
    report("Ensemble Gradient Flow", 0.0 if ens_grad_norm > 0 else 1.0, threshold=0.1)

if __name__ == "__main__":
    torch.manual_seed(42)
    try:
        check_geometric_conservation()
        check_ensemble_diversity()
        check_gradient_flow()
        print("\nAll IO Checks Passed.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")

