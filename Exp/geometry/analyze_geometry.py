import torch
import matplotlib.pyplot as plt
import numpy as np
from swe_sphere_data import SWESphereDataset
from models_geo import CNNBaseline, UniPhyGeoAdapter

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SWESphereDataset()
    x, y, topo = ds[500] 
    x, y, topo = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), topo.unsqueeze(0).to(device)
    
    model_cnn = CNNBaseline().to(device)
    model_geo = UniPhyGeoAdapter(64, 128).to(device)
    model_cnn.load_state_dict(torch.load("cnn_geo.pth"))
    model_geo.load_state_dict(torch.load("uniphy_geo.pth"))
    
    with torch.no_grad():
        p_cnn = model_cnn(x, topo)
        p_geo = model_geo(x, topo)
        
    err_cnn = torch.abs(p_cnn - y).cpu().numpy()[0, 0]
    err_geo = torch.abs(p_geo - y).cpu().numpy()[0, 0]
    topo_np = topo.cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(topo_np, cmap='terrain')
    axes[0].set_title("Topography (The Mountain)")
    plt.colorbar(im0, ax=axes[0])
    
    vmax = max(err_cnn.max(), err_geo.max())
    
    im1 = axes[1].imshow(err_cnn, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title("Standard CNN Error")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(err_geo, cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title("UniPhy (Riemannian) Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.savefig("exp2_geometry_error.png")
    
    mask = topo_np > 0.5
    print(f"Mean Error in Mountainous Region:")
    print(f"CNN: {err_cnn[mask].mean():.6f}")
    print(f"UniPhy: {err_geo[mask].mean():.6f}")

if __name__ == "__main__":
    analyze()

