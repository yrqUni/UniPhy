import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UniPhyOps import TemporalPropagator
from UniPhyFFN import UniPhyFeedForwardNetwork
from ModelUniPhy import UniPhyModel

def check_plu_invertibility():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim).to(device)
    x = torch.randn(16, dim, device=device, dtype=torch.cdouble)
    x_rec = prop.basis.decode(prop.basis.encode(x))
    err = (x - x_rec).abs().max().item()
    if err < 1e-12: pass
    else: print(f"PLU Inversion Error: {err:.2e}")

def check_parapool_conservation():
    dim, expand, num_experts = 64, 4, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn = UniPhyFeedForwardNetwork(dim, expand, num_experts).to(device)
    x = torch.randn(4, dim, 16, 16, device=device, dtype=torch.cdouble)
    delta = ffn(x)
    mean_val = delta.mean(dim=(-2, -1)).abs().max().item()
    if mean_val < 1e-12: pass
    else: print(f"FFN Conservation Error: {mean_val:.2e}")

def check_source_sink_dynamics():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim, noise_scale=0.0).to(device)
    h = torch.randn(1, dim, device=device, dtype=torch.cdouble)
    x_zero = torch.zeros_like(h)
    h_next = prop.forward(h, x_zero, dt=1.0)
    diff = (h_next - h).abs().max().item()
    if diff > 1e-8: pass
    else: print(f"Source-Sink Dynamics Error: {diff:.2e}")

def check_ou_noise_scaling():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim, noise_scale=0.1).to(device)
    prop.train()
    target_shape = (1000, dim)
    noise_small = prop.generate_stochastic_term(target_shape, dt=0.1, dtype=torch.cdouble)
    noise_large = prop.generate_stochastic_term(target_shape, dt=10.0, dtype=torch.cdouble)
    std_small = noise_small.std().item()
    std_large = noise_large.std().item()
    if std_large > std_small: pass
    else: print(f"OU Noise Scaling Error: {std_small:.2e} vs {std_large:.2e}")

def check_semigroup_property():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim, noise_scale=0.0).to(device)
    prop.eval()
    h0 = torch.randn(1, dim, device=device, dtype=torch.cdouble)
    x0 = torch.zeros(1, dim, device=device, dtype=torch.cdouble)
    T_total = 10.0
    h_jump = prop.forward(h0, x0, dt=T_total)
    steps = 100
    dt_small = T_total / steps
    h_step = h0
    for _ in range(steps):
        h_step = prop.forward(h_step, x0, dt=dt_small)
    diff = (h_jump - h_step).abs().max().item()
    if diff < 1e-10: pass
    else: print(f"Semigroup Property Error: {diff:.2e}")

def check_variable_dt_broadcasting():
    dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prop = TemporalPropagator(dim, noise_scale=0.0).to(device)
    prop.eval()
    B_HW, T = 16, 5
    h = torch.randn(B_HW, T, dim, device=device, dtype=torch.cdouble)
    x = torch.zeros(B_HW, T, dim, device=device, dtype=torch.cdouble)
    dt_multi = torch.rand(B_HW, T, device=device) + 0.5
    op_d, op_phi1 = prop.get_transition_operators(dt_multi)
    if op_d.shape == (B_HW, T, dim): pass
    else: print(f"Broadcasting Shape Error: {op_d.shape}")

def check_full_model_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 5, 2, 32, 32
    
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W).to(device)
    model.eval()
    
    for block in model.blocks:
        block.prop.noise_scale = 0.0
    
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.rand(B, T, device=device, dtype=torch.float64) + 1.0
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        
        z = model.encoder(x)
        z_ic = z.clone()
        weights = F.softmax(model.fusion_weights, dim=0)
        z_fused = 0
        
        for i, block in enumerate(model.blocks):
            z = z + z_ic * model.ic_scale
            
            B_seq, T_seq, D_seq, H_seq, W_seq = z.shape
            x_s = z.view(B_seq * T_seq, D_seq, H_seq, W_seq).permute(0, 2, 3, 1)
            x_s = block._complex_norm(x_s, block.norm_spatial).permute(0, 3, 1, 2)
            x_s = block._spatial_op(x_s)
            x_spatial = x_s.view(B_seq, T_seq, D_seq, H_seq, W_seq) + z
            
            x_t = x_spatial.permute(0, 3, 4, 1, 2).reshape(B_seq * H_seq * W_seq, T_seq, D_seq)
            x_t = block._complex_norm(x_t, block.norm_temporal)
            dt_expanded = dt.view(B_seq, 1, 1, T_seq).expand(B_seq, H_seq, W_seq, T_seq).reshape(B_seq * H_seq * W_seq, T_seq)
            
            x_encoded = block.prop.basis.encode(x_t)
            gate = F.silu(block.prop.input_gate(x_encoded.real))
            x_encoded = x_encoded * torch.complex(gate, torch.zeros_like(gate))
            
            op_decay, op_forcing = block.prop.get_transition_operators(dt_expanded, x_encoded)
            bias = block.prop._get_source_bias()
            x_forcing = x_encoded + bias
            u_t = x_forcing * op_forcing
            
            h = torch.zeros(B_seq * H_seq * W_seq, D_seq, device=device, dtype=torch.cdouble)
            h_seq = []
            
            for t in range(T_seq):
                h = h * op_decay[:, t] + u_t[:, t]
                h_seq.append(h)
                
            h_eigen = torch.stack(h_seq, dim=1)
            x_drift = block.prop.basis.decode(h_eigen).real.view(B_seq, H_seq, W_seq, T_seq, D_seq).permute(0, 3, 4, 1, 2)
            x_out = x_drift + x_spatial
            
            x_in_ffn = x_out.view(B_seq * T_seq, D_seq, H_seq, W_seq)
            delta_p = block.ffn(x_in_ffn)
            z = x_out + delta_p.view(B_seq, T_seq, D_seq, H_seq, W_seq)
            
            z_fused = z_fused + z * weights[i]
            
        out_serial = model.decoder(z_fused, x)
        
        diff = (out_parallel - out_serial).abs().max().item()
        if diff < 1e-12: pass
        else: print(f"PScan Consistency Error: {diff:.2e}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    check_plu_invertibility()
    check_parapool_conservation()
    check_source_sink_dynamics()
    check_ou_noise_scaling()
    check_semigroup_property()
    check_variable_dt_broadcasting()
    if torch.cuda.is_available():
        check_full_model_consistency()
        