import torch
import torch.optim as optim
import torch.nn.functional as F
from wave_data import get_wave_loader
from models_physics import DiscreteBaseline, UniPhyPhysicsAdapter

def train():
    device = torch.device("cuda")
    loader = get_wave_loader()
    N = 64
    
    model_base = DiscreteBaseline(N=N).to(device)
    model_uni = UniPhyPhysicsAdapter(N=N).to(device)
    
    opt_base = optim.AdamW(model_base.parameters(), lr=1e-3)
    opt_uni = optim.AdamW(model_uni.parameters(), lr=2e-4, weight_decay=1e-5)
    
    print("Training on dt=1.0 wave propagation (Fixing Dissipation)...")
    for epoch in range(2048):
        for u_t, u_tp1 in loader:
            u_t, u_tp1 = u_t.to(device), u_tp1.to(device)
            dt_t = torch.ones(u_t.shape[0], 1, device=device) * 1.0
            
            opt_base.zero_grad()
            loss_base = F.mse_loss(model_base(u_t), u_tp1)
            loss_base.backward(); opt_base.step()
            
            opt_uni.zero_grad()
            mu = model_uni(u_t, dt_t)
            loss_uni = F.mse_loss(mu, u_tp1)
            
            energy_loss = 0.01 * torch.abs(mu.pow(2).sum() - u_t.pow(2).sum())
            (loss_uni + energy_loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model_uni.parameters(), 0.5)
            
            opt_uni.step()
            
        if epoch % 50 == 0:
            with torch.no_grad():
                out_std = mu.std().item()
            print(f"Epoch {epoch} | Base: {loss_base.item():.5f} | UniPhy: {loss_uni.item():.5f} | Output Std: {out_std:.4f}")

    torch.save(model_base.state_dict(), "base_wave.pth")
    torch.save(model_uni.state_dict(), "uniphy_wave.pth")

if __name__ == "__main__":
    train()

