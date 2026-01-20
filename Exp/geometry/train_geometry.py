import torch
import torch.optim as optim
import torch.nn.functional as F
from swe_sphere_data import get_geo_data
from models_geo import CNNBaseline, UniPhyGeoAdapter

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_geo_data()
    
    h, w = 64, 128
    model_cnn = CNNBaseline().to(device)
    model_geo = UniPhyGeoAdapter(h, w).to(device)
    
    opt_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3)
    opt_geo = optim.Adam(model_geo.parameters(), lr=1e-3)
    
    for epoch in range(100):
        for x, y, topo in loader:
            x, y, topo = x.to(device), y.to(device), topo.to(device)
            
            p_cnn = model_cnn(x, topo)
            loss_cnn = F.mse_loss(p_cnn, y)
            opt_cnn.zero_grad(); loss_cnn.backward(); opt_cnn.step()
            
            p_geo = model_geo(x, topo)
            loss_geo = F.mse_loss(p_geo, y)
            opt_geo.zero_grad(); loss_geo.backward(); opt_geo.step()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | CNN Loss: {loss_cnn.item():.6f} | Geo Loss: {loss_geo.item():.6f}")

    torch.save(model_cnn.state_dict(), "cnn_geo.pth")
    torch.save(model_geo.state_dict(), "uniphy_geo.pth")

if __name__ == "__main__":
    train()

