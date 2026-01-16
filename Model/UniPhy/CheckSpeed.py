import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from SpectralStepTriton import SpectralStep as SpectralStepTriton

class SpectralStepPyTorch(nn.Module):
    def __init__(self, in_ch, rank=32, w_freq=64):
        super().__init__()
        self.in_ch = in_ch
        self.rank = rank
        self.w_freq = w_freq
        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, w_freq)),
            nn.Conv2d(in_ch, rank * 2, 1)
        )
        nn.init.uniform_(self.estimator[-1].weight, -0.01, 0.01)

    def forward(self, x, dt):
        B, C, H, W = x.shape
        x_spec = torch.fft.rfft2(x, norm="ortho")
        params = self.estimator(x)
        nu, theta = torch.chunk(params, 2, dim=1)
        nu = F.softplus(nu)
        theta = torch.tanh(theta) * math.pi

        if dt.dim() == 0: dt = dt.view(1).expand(B)
        elif dt.dim() > 1: dt = dt.view(B)
        dt_view = dt.view(B, 1, 1, 1)

        decay = torch.exp(-nu * dt_view)
        angle = theta * dt_view
        
        filter_real = (decay * torch.cos(angle)).sum(dim=1)
        filter_imag = (decay * torch.sin(angle)).sum(dim=1)
        
        filter_complex = torch.complex(filter_real, filter_imag).unsqueeze(1)
        
        target_spec = x_spec[:, :, :, :self.w_freq]
        out_spec = target_spec * filter_complex
        
        x_final_spec = torch.zeros_like(x_spec)
        x_final_spec[:, :, :, :self.w_freq] = out_spec
        x_final_spec[:, :, :, self.w_freq:] = x_spec[:, :, :, self.w_freq:]
        
        return torch.fft.irfft2(x_final_spec, s=(H, W), norm="ortho")

def measure(model, x, dt, steps=50):
    torch.cuda.synchronize()
    try:
        for _ in range(5):
            y = model(x, dt)
            loss = y.sum()
            loss.backward()
    except Exception as e:
        return float('inf'), 0
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    
    start_event.record()
    for _ in range(steps):
        y = model(x, dt)
        loss = y.sum()
        loss.backward()
    end_event.record()
    torch.cuda.synchronize()
    
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return start_event.elapsed_time(end_event) / steps, max_mem

def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    
    configs = [
        (4, 64, 64, 64),
        (8, 128, 128, 128),
        (16, 256, 256, 256)
    ]

    print(f"{'Config':<20} | {'Impl':<8} | {'Time(ms)':<10} | {'Mem(MB)':<10} | {'Speedup':<8}")
    print("-" * 75)

    for B, C, H, W in configs:
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)
        dt = torch.rand(B, device=device, requires_grad=True)
        
        pt_model = SpectralStepPyTorch(C, rank=32, w_freq=W//2+1).to(device)
        tr_model = SpectralStepTriton(C, rank=32, w_freq=W//2+1).to(device)
        
        t_pt, m_pt = measure(pt_model, x, dt)
        t_tr, m_tr = measure(tr_model, x, dt)
        
        speedup = t_pt / t_tr if t_tr > 0 else 0
        
        print(f"({B},{C},{H},{W:<3}) | {'PyTorch':<8} | {t_pt:<10.2f} | {m_pt:<10.2f} | 1.00x")
        print(f"{'':<20} | {'Triton':<8} | {t_tr:<10.2f} | {m_tr:<10.2f} | {speedup:.2f}x")
        print("-" * 75)

if __name__ == "__main__":
    main()

