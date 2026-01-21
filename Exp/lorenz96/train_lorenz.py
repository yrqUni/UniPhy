import torch
import torch.optim as optim
import torch.nn.functional as F
from lorenz_data import get_data
from models import DeterministicBaseline, UniPhyLorenzAdapter 

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, test_data, dt = get_data(T=2000)
    train_x, train_y = train_data
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    
    model_det = DeterministicBaseline(N=40).to(device)
    model_sto = UniPhyLorenzAdapter(N=40).to(device) 

    print(f"🔥 UniPhy Adapter Loaded. Parameters: {sum(p.numel() for p in model_sto.parameters())}")
    
    opt_det = optim.Adam(model_det.parameters(), lr=1e-3)
    opt_sto = optim.Adam(model_sto.parameters(), lr=5e-4)
    
    epochs = 10000
    batch_size = 128
    
    print("Start Training...")
    for epoch in range(epochs):
        idx = torch.randint(0, len(train_x), (batch_size,))
        bx, by = train_x[idx], train_y[idx]
        
        pred_det = model_det(bx)
        loss_det = F.mse_loss(pred_det, by)
        
        opt_det.zero_grad()
        loss_det.backward()
        opt_det.step()
        
        mu, logvar = model_sto(bx)
        loss_sto = F.gaussian_nll_loss(mu, by, torch.exp(logvar))
        
        opt_sto.zero_grad()
        loss_sto.backward()
        opt_sto.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Det Loss: {loss_det.item():.6f} | UniPhy Loss: {loss_sto.item():.6f}")

    torch.save(model_det.state_dict(), 'det.pth')
    torch.save(model_sto.state_dict(), 'sto.pth')
    print("Training Done.")

if __name__ == "__main__":
    train()

