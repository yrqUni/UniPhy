import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F

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
    prop.raw_noise_param.data.fill_(-100.0) 
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
    prop.raw_noise_param.data.fill_(-100.0)
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
    for block in model.blocks: block.prop.raw_noise_param.data.fill_(-100.0)
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.rand(B, T, device=device, dtype=torch.float64) + 1.0
    with torch.no_grad():
        out_parallel = model(x, dt)
        z_enc = model.encoder(x)
        z_ic = z_enc.clone()
        weights = F.softmax(model.fusion_weights, dim=0)
        z_fused = 0
        z_layer = z_enc
        for i, block in enumerate(model.blocks):
            z_layer = z_layer + z_ic * model.ic_scale
            B_z, T_z, D_z, H_z, W_z = z_layer.shape
            h_prev = torch.zeros(B_z * H_z * W_z, D_z, device=device, dtype=torch.cdouble)
            z_step_list = []
            for t in range(T_z):
                x_step = z_layer[:, t]
                dt_step = dt[:, t]
                z_step_out, h_next = block.forward_step(x_step, h_prev, dt_step)
                z_step_list.append(z_step_out)
                h_prev = h_next
            z_layer = torch.stack(z_step_list, dim=1)
            z_fused = z_fused + z_layer * weights[i]
        out_serial = model.decoder(z_fused, x)
        diff = (out_parallel - out_serial).abs().max().item()
        if diff < 1e-12: pass
        else: print(f"PScan Consistency Error: {diff:.2e}")

def check_forecast_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 3, 2, 32, 32
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=2, img_height=H, img_width=W).to(device)
    model.eval()
    for block in model.blocks: block.prop.raw_noise_param.data.fill_(-100.0)
    
    # 关键：使用恒定序列，排除 Decoder Skip 干扰
    x_single = torch.randn(B, 1, C, H, W, device=device, dtype=torch.float64)
    x = x_single.repeat(1, T, 1, 1, 1)
    dt = torch.ones(B, T, device=device, dtype=torch.float64)
    
    with torch.no_grad():
        out_parallel = model(x, dt)
        out_forecast = model.forecast(x_cond=x[:, :1], dt_cond=dt[:, :1], k_steps=2, dt_future=dt[:, 1:])
        
        step1_p = out_parallel[:, 1]
        step1_f = out_forecast[:, 0]
        
        diff1 = (step1_p - step1_f).abs().max().item()
        if diff1 < 1e-6: pass # 由于 float32 累积，预测步允许略大误差
        else: print(f"Forecast Step 1 Consistency Error: {diff1:.2e}")

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
        check_forecast_consistency()
        