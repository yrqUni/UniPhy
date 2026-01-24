import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import TemporalPropagator, ComplexSVDTransform
from UniPhyFFN import UniPhyFeedForwardNetwork
from ModelUniPhy import UniPhyModel
from UniPhyIO import UniPhyEncoder, UniPhyEnsembleDecoder

def check_basis_invertibility():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis = ComplexSVDTransform(dim).to(device)
    x = torch.randn(16, dim, device=device, dtype=torch.cdouble)
    x_enc = basis.encode(x)
    x_dec = basis.decode(x_enc)
    err = (x - x_dec).abs().max().item()
    if err < 1e-12: pass
    else: print(f"Basis Inversion Error: {err:.2e}")

def check_eigenvalue_stability():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim).to(device)
    lambda_val = prop._get_effective_lambda()
    max_real = lambda_val.real.max().item()
    if max_real <= 1e-6: pass
    else: print(f"Eigenvalue Stability Error: Max Real Part {max_real:.2e} > 0")

def check_ffn_causality():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D, H, W = 1, 4, 16, 8, 8
    ffn = UniPhyFeedForwardNetwork(dim=D, expand=4, num_experts=4).to(device)
    ffn.eval()
    
    x = torch.randn(B, T, D, H, W, device=device, dtype=torch.cdouble)
    
    x_par = x.reshape(B * T, D, H, W)
    out_par = ffn(x_par).reshape(B, T, D, H, W)
    
    out_ser_list = []
    for t in range(T):
        out_ser_list.append(ffn(x[:, t]))
    out_ser = torch.stack(out_ser_list, dim=1)
    
    diff = (out_par - out_ser).abs().max().item()
    if diff < 1e-12: pass
    else: print(f"FFN Causality Error: {diff:.2e}")

def check_io_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 1, 5
    C_in, C_out = 2, 2
    H, W = 721, 1440 
    Patch, Embed = 16, 64
    
    enc = UniPhyEncoder(in_ch=C_in, embed_dim=Embed, patch_size=Patch, img_height=H, img_width=W).to(device)
    dec = UniPhyEnsembleDecoder(out_ch=C_out, latent_dim=Embed, patch_size=Patch, img_height=H, img_width=W).to(device)
    
    x = torch.randn(B, T, C_in, H, W, device=device, dtype=torch.float64)
    z = enc(x)
    
    expected_h = (H + (Patch - H % Patch) % Patch) // Patch
    expected_w = (W + (Patch - W % Patch) % Patch) // Patch
    assert z.shape[-2] == expected_h
    assert z.shape[-1] == expected_w
    
    out = dec(z)
    assert out.shape == (B, T, C_out, H, W)

def check_full_model_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Full Model Consistency on {device}...")
    
    B, T, C, H, W = 1, 5, 2, 33, 33
    dt_ref = 6.0
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W, dt_ref=dt_ref, noise_scale=0.0).to(device)
    model.eval()
    
    for block in model.blocks:
        block.prop.noise_scale = 0.0
    
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.ones(B, T, device=device, dtype=torch.float64) * dt_ref
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        
        z = model.encoder(x)
        B_z, T_z, D_z, H_z, W_z = z.shape
        
        for block in model.blocks:
            h_state = torch.zeros(B_z * H_z * W_z, 1, block.dim, dtype=torch.cdouble, device=device)
            flux_state = torch.zeros(B_z, block.dim, dtype=torch.cdouble, device=device)
            z_steps = []
            
            for t in range(T):
                x_step = z[:, t]
                dt_step = dt[:, t]
                
                z_next, h_next, flux_next = block.forward_step(x_step, h_state, dt_step, flux_state)
                
                z_steps.append(z_next)
                h_state = h_next
                flux_state = flux_next
            
            z = torch.stack(z_steps, dim=1)
            
        out_serial = model.decoder(z)
        
        diff = (out_parallel - out_serial).abs().max().item()
        
    if diff < 1e-7:
        print(f"Consistency Check Passed. Diff: {diff:.2e}")
    else:
        print(f"Consistency Check FAILED. Diff: {diff:.2e}")
        print(f"   Parallel Mean: {out_parallel.mean():.6f}")
        print(f"   Serial   Mean: {out_serial.mean():.6f}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("Running Checks...")
    check_basis_invertibility()
    check_eigenvalue_stability()
    check_ffn_causality()
    check_io_shapes()
    if torch.cuda.is_available():
        check_full_model_consistency()
    print("Checks Completed.")
