import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from types import SimpleNamespace
from tqdm import tqdm

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel
from ERA5 import ERA5_Dataset

def get_args():
    return SimpleNamespace(
        input_shape=(721, 1440),
        in_channels=2,
        out_channels=2,
        embed_dim=300,
        patch_size=16,
        depth=6,
        img_height=721,
        img_width=1440,
        dropout=0.0,
        ensemble_size=8,
        dt_ref=6.0,
        data_root="/nfs/ERA5_data/data_norm",
        year_range=[2017, 2021],
        ctx_len=1,
        pred_len=4,
        checkpoint_path="./uniphy_ckpt/latest.pth",
        save_path="result.gif",
        use_tf32=True
    )

def render_frame(t, gt, pred_mean, pred_std, sample1, sample2):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    cmap = 'jet'
    v_min, v_max = gt.min(), gt.max()

    ax = axes[0, 0]
    im = ax.imshow(gt, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title(f"Ground Truth (t={t+1})")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[0, 1]
    im = ax.imshow(pred_mean, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Ensemble Mean")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[0, 2]
    im = ax.imshow(np.abs(gt - pred_mean), cmap='inferno')
    ax.set_title("Absolute Error")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 0]
    im = ax.imshow(sample1, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("SDE Member 1")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 1]
    im = ax.imshow(sample2, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("SDE Member 2")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 2]
    im = ax.imshow(pred_std, cmap='viridis')
    ax.set_title("Uncertainty (Std Dev)")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def main():
    plt.switch_backend('Agg')
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = UniPhyModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        dim=args.embed_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        img_height=args.img_height,
        img_width=args.img_width,
        dropout=args.dropout,
        ensemble_size=args.ensemble_size
    ).to(device)

    if os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        sd = ckpt.get('model', ckpt)
        nsd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(nsd, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found. Using random weights.")

    model.eval()

    t_total = args.ctx_len + args.pred_len
    try:
        ds = ERA5_Dataset(
            input_dir=args.data_root, year_range=args.year_range, is_train=False,
            sample_len=t_total, eval_sample=1, max_cache_size=1, rank=0, gpus=1
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        full_seq = next(iter(loader))
    except Exception as e:
        print(f"Data loading failed: {e}. Using random data.")
        full_seq = torch.randn(1, t_total, args.in_channels, args.img_height, args.img_width)

    full_seq = full_seq.to(device).float()
    initial_state = full_seq[:, args.ctx_len-1 : args.ctx_len]
    gt_future = full_seq[:, args.ctx_len:]

    ensemble_trajectories = []

    for m in tqdm(range(args.ensemble_size), desc="Ensemble Members"):
        current_state = initial_state.clone()
        trajectory = []

        for t in range(args.pred_len):
            dt = torch.full((1, 1), args.dt_ref, device=device)
            with torch.no_grad():
                next_state = model(current_state, dt)
            trajectory.append(next_state.cpu().numpy())
            current_state = next_state

        trajectory = np.concatenate(trajectory, axis=1)
        ensemble_trajectories.append(trajectory)

    ensemble_stack = np.concatenate(ensemble_trajectories, axis=0)

    p_mean = np.mean(ensemble_stack, axis=0)
    p_std = np.std(ensemble_stack, axis=0)
    gt_np = gt_future.cpu().numpy()[0]

    sample1 = ensemble_stack[0]
    sample2 = ensemble_stack[min(1, args.ensemble_size-1)]

    frames = []
    c_idx = 0

    for t in range(args.pred_len):
        frame = render_frame(
            t,
            gt=gt_np[t, c_idx],
            pred_mean=p_mean[t, c_idx],
            pred_std=p_std[t, c_idx],
            sample1=sample1[t, c_idx],
            sample2=sample2[t, c_idx]
        )
        frames.append(frame)

    imageio.mimsave(args.save_path, frames, fps=2)

if __name__ == "__main__":
    main()

