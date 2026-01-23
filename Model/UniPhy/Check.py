import torch
import torch.nn as nn
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
    op_d, op_f = prop.get_transition_operators(dt_multi, x_state=x)
    if op_d.shape == (B_HW, T, dim): pass
    else: print(f"Broadcasting Shape Error: {op_d.shape}")

def check_full_model_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, C, H, W = 1, 5, 2, 32, 32
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=16, depth=1, img_height=H, img_width=W).to(device)
    model.eval()
    x = torch.randn(B, T, C, H, W, device=device, dtype=torch.float64)
    dt = torch.rand(B, T, device=device, dtype=torch.float64) + 1.0
    with torch.no_grad():
        out_parallel = model(x, dt)
        z = model.encoder(x)
        block = model.blocks[0]
        B_z, T_z, D_z, H_z, W_z = z.shape
        h_curr = torch.zeros(B_z * H_z * W_z, D_z, device=device, dtype=torch.cdouble)
        z_seq_list = []
        for t in range(T_z):
            x_step = z[:, t]
            dt_step = dt[:, t]
            z_next, h_next = block.forward_step(x_step, h_curr, dt_step)
            z_seq_list.append(z_next)
            h_curr = h_next
        z_seq = torch.stack(z_seq_list, dim=1)
        out_seq = model.decoder(z_seq, x)
        diff = (out_parallel - out_seq).abs().max().item()
        if diff < 1e-10: pass
        else: print(f"PScan Consistency Error: {diff:.2e}")

def check_forecast_implementation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T_cond, K, C, H, W = 2, 5, 3, 2, 32, 32
    model = UniPhyModel(in_channels=C, out_channels=C, embed_dim=32, depth=2, img_height=H, img_width=W).to(device)
    model.eval()
    x_cond = torch.randn(B, T_cond, C, H, W, device=device, dtype=torch.float64)
    dt_cond = torch.ones(B, T_cond, device=device, dtype=torch.float64)
    dt_future = torch.ones(B, K, device=device, dtype=torch.float64) * 0.5
    pred = model.forecast(x_cond, dt_cond, K, dt_future)
    expected_shape = (B, K, C, H, W)
    if pred.shape == expected_shape: pass
    else: print(f"Forecast Shape Error: Expected {expected_shape}, got {pred.shape}")
    if torch.isnan(pred).any() or torch.isinf(pred).any(): print("Forecast contains NaN or Inf")
        
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
        check_forecast_implementation()
