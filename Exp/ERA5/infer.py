import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Model/UniPhy")))
try:
    from ModelUniPhy import UniPhy
except ImportError:
    print("Error: ModelUniPhy not found.")
    sys.exit(1)

try:
    from ERA5 import ERA5_Dataset
except ImportError:
    print("Error: ERA5_Dataset not found.")
    sys.exit(1)

def get_args():
    return SimpleNamespace(
        input_ch=2,
        out_ch=2,
        input_size=(64, 64), 
        emb_ch=64,
        hidden_factor=(2, 2),
        ConvType="conv",
        Arch="unet",
        dist_mode="gaussian",
        convlru_num_blocks=2,
        down_mode="avg",
        ffn_ratio=2.0,
        lru_rank=64,
        koopman_use_noise=False,
        koopman_noise_scale=1.0,
        dt_ref=1.0,
        inj_k=2.0,
        dynamics_mode="advection",
        spectral_modes_h=12,
        spectral_modes_w=12,
        interpolation_mode="bilinear",
        pscan_use_decay=True,
        pscan_use_residual=True,
        pscan_chunk_size=32,
        data_path="./Data", 
        val_batch_size=1,
        num_workers=4,
        T=4, 
        forward_steps=20, 
        checkpoint_path="checkpoint.pth",
        save_path="result.gif"
    )

def denorm_numpy(x, mean, std):
    return x * std + mean

def render_frame(t, gt, pred_mean, pred_std, sample1, sample2, channel_idx=0):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    cmap = 'jet'
    
    v_min = gt.min()
    v_max = gt.max()

    ax = axes[0, 0]
    im = ax.imshow(gt[channel_idx], cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title(f"Ground Truth (t={t+1})")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[0, 1]
    im = ax.imshow(pred_mean[channel_idx], cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Prediction Mean (Deterministic)")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[0, 2]
    error = np.abs(gt[channel_idx] - pred_mean[channel_idx])
    im = ax.imshow(error, cmap='inferno')
    ax.set_title("Absolute Error |GT - Mean|")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 0]
    im = ax.imshow(sample1[channel_idx], cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Stochastic Sample 1")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 1]
    im = ax.imshow(sample2[channel_idx], cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Stochastic Sample 2")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 2]
    im = ax.imshow(pred_std[channel_idx], cmap='viridis')
    ax.set_title("Predicted Uncertainty (Sigma)")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = UniPhy(args).to(device)
    
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found. Using random weights.")

    model.eval()

    try:
        dataset = ERA5_Dataset(args, split="test")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        data_iter = iter(dataloader)
        full_seq, _ = next(data_iter) 
    except Exception as e:
        print(f"Dataset load failed ({e}), creating dummy data.")
        full_seq = torch.randn(1, args.T + args.forward_steps, args.input_ch, *args.input_size)

    full_seq = full_seq.to(device)
    B, total_len, C, H, W = full_seq.shape
    
    cond_len = args.T
    pred_len = args.forward_steps
    
    input_seq = full_seq[:, :cond_len]
    gt_seq = full_seq[:, cond_len : cond_len + pred_len]

    print("Running Deterministic Inference...")
    with torch.no_grad():
        out_det_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            sample=False 
        )
    
    out_det = out_det_cpu.numpy() 
    mu_det = out_det[:, :, :args.out_ch]
    sigma_det = out_det[:, :, args.out_ch:]

    print("Running Stochastic Inference (Sample 1)...")
    with torch.no_grad():
        out_s1_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            sample=True
        )
    sample1 = out_s1_cpu.numpy()[:, :, :args.out_ch]

    print("Running Stochastic Inference (Sample 2)...")
    with torch.no_grad():
        out_s2_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            sample=True
        )
    sample2 = out_s2_cpu.numpy()[:, :, :args.out_ch]

    gt_np = gt_seq.cpu().numpy()
    
    print("Generating GIF...")
    frames = []
    for t in range(pred_len):
        frame = render_frame(
            t,
            gt=gt_np[0, t],
            pred_mean=mu_det[0, t],
            pred_std=sigma_det[0, t],
            sample1=sample1[0, t],
            sample2=sample2[0, t],
            channel_idx=0 
        )
        frames.append(frame)

    imageio.mimsave(args.save_path, frames, fps=5)
    print(f"Saved visualization to {args.save_path}")

if __name__ == "__main__":
    main()

