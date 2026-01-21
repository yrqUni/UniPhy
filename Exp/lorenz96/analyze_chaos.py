import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lorenz_data import get_data
from models import DeterministicBaseline, UniPhyLorenzAdapter

def compute_crps(ens, gt):
    M = ens.shape[0]
    mae = np.mean(np.abs(ens - gt[None, ...]), axis=0)
    ens_sorted = np.sort(ens, axis=0)
    ranks = np.arange(1, M + 1).reshape(M, 1, 1)
    spread = np.mean(ens_sorted * (2 * ranks - M - 1), axis=0) / M
    return np.mean(mae - spread)

def compute_psd(traj):
    fft_vals = np.fft.rfft(traj, axis=1)
    psd = np.abs(fft_vals)**2
    return np.mean(psd, axis=0)

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_data, _ = get_data(T=2000)
    x0 = test_data[0][0].unsqueeze(0).to(device) 
    T_pred = 1000 
    
    model_det = DeterministicBaseline(N=40).to(device)
    model_sto = UniPhyLorenzAdapter(N=40).to(device)
    model_det.load_state_dict(torch.load('det.pth', map_location=device))
    model_sto.load_state_dict(torch.load('sto.pth', map_location=device))
    model_det.eval(); model_sto.eval()

    print("Running Rollouts...")
    traj_gt = test_data[0][:T_pred+1].numpy()
    
    pred_det = [x0]
    curr = x0
    with torch.no_grad():
        for _ in range(T_pred):
            curr = model_det(curr)
            pred_det.append(curr)
    traj_det = torch.cat(pred_det).cpu().numpy()
    
    n_ens = 10
    ens_list = []
    for i in range(n_ens):
        curr = x0
        this_traj = [x0]
        with torch.no_grad():
            for _ in range(T_pred):
                curr = model_sto.sample(curr)
                curr = torch.clamp(curr, -20, 20)
                this_traj.append(curr)
        ens_list.append(torch.cat(this_traj).cpu().numpy())
    ens_trajs = np.array(ens_list) 
    ens_mean = np.mean(ens_trajs, axis=0)

    rmse_det = np.sqrt(np.mean((traj_det - traj_gt)**2))
    rmse_uni = np.sqrt(np.mean((ens_mean - traj_gt)**2))
    crps_uni = compute_crps(ens_trajs, traj_gt)

    print(f"\n{'='*30}")
    print(f"METRICS REPORT")
    print(f"{'-'*30}")
    print(f"Baseline RMSE: {rmse_det:.4f}")
    print(f"UniPhy Mean RMSE: {rmse_uni:.4f}")
    print(f"UniPhy Ensemble CRPS: {crps_uni:.4f} (Lower is better)")
    print(f"{'='*30}\n")

    print("Plotting Energy Spectrum...")
    psd_gt = compute_psd(traj_gt)
    psd_det = compute_psd(traj_det)
    psd_uni = compute_psd(ens_trajs[0]) 

    plt.figure(figsize=(8, 6))
    freqs = np.arange(len(psd_gt))
    plt.loglog(freqs, psd_gt, 'k-', label='Ground Truth', lw=2)
    plt.loglog(freqs, psd_det, 'r--', label='Baseline (Spectral Collapse)', lw=2)
    plt.loglog(freqs, psd_uni, 'b-', label='UniPhy (Stochastic Detail)', alpha=0.7)
    
    plt.title("Spatial Energy Spectrum (Lorenz-96)")
    plt.xlabel("Wavenumber"); plt.ylabel("Energy Density")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("exp1_energy_spectrum.png", dpi=300)

    plt.figure(figsize=(18, 6))
    for i, (t, title, c) in enumerate([(traj_gt, "GT", "black"), (traj_det, "Baseline", "red"), (ens_trajs[0], "UniPhy", "blue")]):
        plt.subplot(1, 3, i+1)
        plt.plot(t[:, 0], t[:, 1], lw=0.5, color=c)
        plt.title(title); plt.xlabel("X0"); plt.ylabel("X1")
    plt.tight_layout()
    plt.savefig("exp1_chaos_attractor.png", dpi=300)

    plt.figure(figsize=(10, 5))
    sns.kdeplot(traj_gt.flatten(), label='Ground Truth', fill=True, color='gray', alpha=0.3)
    sns.kdeplot(traj_det.flatten(), label='Baseline', color='red')
    sns.kdeplot(ens_trajs.flatten(), label='UniPhy Ensemble', color='blue')
    plt.title("PDF Match"); plt.legend()
    plt.savefig("exp1_spectral_bias.png")

    print("✅ All plots saved.")

if __name__ == "__main__":
    analyze()

