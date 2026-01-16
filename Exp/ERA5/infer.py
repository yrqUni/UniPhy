import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from types import SimpleNamespace
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Model/UniPhy")))
sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel
from ERA5 import ERA5_Dataset

def get_args():
    return SimpleNamespace(
        input_shape=(721, 1440),
        in_channels=30,
        dim=1536,
        patch_size=4,
        num_layers=24,
        para_pool_expansion=4,
        conserve_energy=True,

        dt_ref=1.0,
        data_root="/nfs/ERA5_data/data_norm",
        year_range=[2017, 2021],
        
        ctx_len=4,
        pred_len=4,
        ensemble_size=8,
        diffusion_steps=20,

        checkpoint_path="./uniphy_ckpt/latest.pth",
        save_path="result.gif",
        
        use_tf32=True
    )

def render_frame(t, gt, pred_mean, pred_std, sample1, sample2, channel_idx=0):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    cmap = 'jet'
    
    v_min = gt.min()
    v_max = gt.max()

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
    error = np.abs(gt - pred_mean)
    im = ax.imshow(error, cmap='inferno')
    ax.set_title("Absolute Error")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 0]
    im = ax.imshow(sample1, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Member 1")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 1]
    im = ax.imshow(sample2, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Member 2")
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[1, 2]
    im = ax.imshow(pred_std, cmap='viridis')
    ax.set_title("Uncertainty")
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
    print(f"Running on {device}")
    
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Building Model...")
    model = UniPhyModel(
        input_shape=args.input_shape,
        in_channels=args.in_channels,
        dim=args.dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        para_pool_expansion=args.para_pool_expansion,
        conserve_energy=args.conserve_energy
    ).to(device)

    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        state_dict = checkpoint.get('model', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {args.checkpoint_path} not found! Using random weights.")

    model.eval()

    total_len = args.ctx_len + args.pred_len
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
        full_seq = torch.randn(1, total_len, args.in_channels, *args.input_shape)

    full_seq = full_seq.to(device).float()
    
    input_seq = full_seq[:, :args.ctx_len]
    gt_seq = full_seq[:, args.ctx_len:]
    
    B, _, C, H, W = input_seq.shape
    
    ctx_dt = torch.full((B, args.ctx_len), args.dt_ref, device=device)
    future_dt = args.dt_ref

    print(f"Running Ensemble Inference ({args.ensemble_size} members)...")
    
    ensemble_preds = []
    
    for i in tqdm(range(args.ensemble_size)):
        pred = model.inference(
            context_x=input_seq,
            context_dt=ctx_dt,
            future_steps=args.pred_len,
            future_dt=future_dt,
            diffusion_steps=args.diffusion_steps
        )
        ensemble_preds.append(pred.cpu().numpy())

    ensemble_stack = np.stack(ensemble_preds, axis=0)
    
    pred_mean = np.mean(ensemble_stack, axis=0)
    pred_std = np.std(ensemble_stack, axis=0)
    
    sample1 = ensemble_stack[0]
    sample2 = ensemble_stack[1] if args.ensemble_size > 1 else ensemble_stack[0]
    gt_np = gt_seq.cpu().numpy()

    print("Generating GIF...")
    frames = []
    
    viz_ch = 0 
    
    for t in range(args.pred_len):
        frame = render_frame(
            t,
            gt=gt_np[0, t, viz_ch],
            pred_mean=pred_mean[0, t, viz_ch],
            pred_std=pred_std[0, t, viz_ch],
            sample1=sample1[0, t, viz_ch],
            sample2=sample2[0, t, viz_ch],
            channel_idx=viz_ch
        )
        frames.append(frame)

    imageio.mimsave(args.save_path, frames, fps=2)
    print(f"Saved visualization to {args.save_path}")

if __name__ == "__main__":
    main()

