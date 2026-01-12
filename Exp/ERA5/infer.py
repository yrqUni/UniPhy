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
        input_size=(721, 1440),
        input_ch=30,
        out_ch=30,
        hidden_factor=(7, 12),
        emb_ch=64,
        convlru_num_blocks=6,
        lru_rank=64,
        down_mode="shuffle",
        dist_mode="gaussian",
        ffn_ratio=1.5,
        ConvType="dcn",
        Arch="unet",
        koopman_use_noise=True,
        koopman_noise_scale=1.0,
        dt_ref=1.0,
        inj_k=2.0,
        max_velocity=5.0,
        dynamics_mode="advection",
        interpolation_mode="bilinear",
        spectral_modes_h=12,
        spectral_modes_w=12,
        data_root="/nfs/ERA5_data/data_org",
        year_range=[2017, 2021],
        T=6, 
        forward_steps=4,
        checkpoint_path="./uniphy_base/ckpt/latest.pth",
        save_path="result.gif",
        use_tf32=False,
        use_compile=False
    )

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
        state_dict = checkpoint.get('model', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found. Using random weights.")

    model.eval()

    cond_len = args.T
    pred_len = args.forward_steps
    total_len = cond_len + pred_len

    try:
        dataset = ERA5_Dataset(
            input_dir=args.data_root,
            year_range=args.year_range,
            is_train=False,
            sample_len=total_len,
            eval_sample=1,
            max_cache_size=1,
            rank=0,
            gpus=1
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        data_iter = iter(dataloader)
        full_seq = next(data_iter)
    except Exception as e:
        print(f"Dataset load failed ({e}), creating dummy data.")
        full_seq = torch.randn(1, total_len, args.input_ch, *args.input_size)

    full_seq = full_seq.to(device)
    B, _, C, H, W = full_seq.shape

    input_seq = full_seq[:, :cond_len]
    gt_seq = full_seq[:, cond_len:]

    revin_stats = model.revin.stats(input_seq)
    
    listT_cond = torch.full((B, cond_len), float(args.dt_ref), device=device)
    listT_future = torch.full((B, pred_len), float(args.dt_ref), device=device)

    print("Running Deterministic Inference...")
    with torch.no_grad():
        out_det_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            listT=listT_cond,
            listT_future=listT_future,
            revin_stats=revin_stats,
            sample=False
        )

    out_det = out_det_cpu.numpy()
    
    if out_det.shape[2] == args.out_ch * 2:
        mu_det = out_det[:, :, :args.out_ch]
        sigma_det = out_det[:, :, args.out_ch:]
    else:
        mu_det = out_det
        sigma_det = np.zeros_like(mu_det)

    print("Running Stochastic Inference (Sample 1)...")
    with torch.no_grad():
        out_s1_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            listT=listT_cond,
            listT_future=listT_future,
            revin_stats=revin_stats,
            sample=True
        )
    sample1 = out_s1_cpu.numpy()
    if sample1.shape[2] == args.out_ch * 2:
         sample1 = sample1[:, :, :args.out_ch]

    print("Running Stochastic Inference (Sample 2)...")
    with torch.no_grad():
        out_s2_cpu, _ = model(
            input_seq,
            mode="i",
            out_gen_num=pred_len,
            listT=listT_cond,
            listT_future=listT_future,
            revin_stats=revin_stats,
            sample=True
        )
    sample2 = out_s2_cpu.numpy()
    if sample2.shape[2] == args.out_ch * 2:
         sample2 = sample2[:, :, :args.out_ch]

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

