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
        decoder_type="ensemble",
        ensemble_size=8,
        diffusion_steps=20,
        dt_ref=1.0,
        data_root="/nfs/ERA5_data/data_norm",
        year_range=[2017, 2021],
        ctx_len=4,
        pred_len=4,
        checkpoint_path="./uniphy_ckpt/latest.pth",
        save_path="result.gif",
        use_tf32=True
    )

def render_frame(t, gt, pred_mean, pred_std, sample1, sample2, channel_idx=0):
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    cmap = 'jet'
    v_min, v_max = gt.min(), gt.max()
    ax = axes[0, 0]
    im = ax.imshow(gt, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title(f"GT (t={t+1})")
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax = axes[0, 1]
    im = ax.imshow(pred_mean, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.set_title("Mean")
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax = axes[0, 2]
    im = ax.imshow(np.abs(gt - pred_mean), cmap='inferno')
    ax.set_title("Error")
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
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = UniPhyModel(
        input_shape=args.input_shape,
        in_channels=args.in_channels,
        dim=args.dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        para_pool_expansion=args.para_pool_expansion,
        conserve_energy=args.conserve_energy,
        decoder_type=args.decoder_type,
        ensemble_size=args.ensemble_size
    ).to(device)
    if os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        sd = ckpt.get('model', ckpt)
        nsd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(nsd, strict=False)
    model.eval()
    t_len = args.ctx_len + args.pred_len
    try:
        ds = ERA5_Dataset(
            input_dir=args.data_root, year_range=args.year_range, is_train=False,
            sample_len=t_len, eval_sample=1, max_cache_size=1, rank=0, gpus=1
        )
        full_seq = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
    except:
        full_seq = torch.randn(1, t_len, args.in_channels, *args.input_shape)
    full_seq = full_seq.to(device).float()
    input_seq = full_seq[:, :args.ctx_len]
    gt_seq = full_seq[:, args.ctx_len:]
    with torch.no_grad():
        preds = model.inference(
            context_x=input_seq,
            context_dt=torch.full((1, args.ctx_len), args.dt_ref, device=device),
            future_steps=args.pred_len,
            future_dt=args.dt_ref,
            diffusion_steps=args.diffusion_steps,
            num_ensemble=args.ensemble_size
        )
    p_np = preds.cpu().numpy()
    gt_np = gt_seq.cpu().numpy()
    if args.decoder_type == "ensemble":
        e_stack = np.transpose(p_np, (2, 0, 1, 3, 4, 5))
    else:
        e_stack = np.expand_dims(p_np, axis=0)
    p_mean = np.mean(e_stack, axis=0)
    p_std = np.std(e_stack, axis=0)
    s1 = e_stack[0]
    s2 = e_stack[min(1, len(e_stack)-1)]
    frames = []
    for t in range(args.pred_len):
        frame = render_frame(
            t,
            gt=gt_np[0, t, 0],
            pred_mean=p_mean[0, t, 0],
            pred_std=p_std[0, t, 0],
            sample1=s1[0, t, 0],
            sample2=s2[0, t, 0]
        )
        frames.append(frame)
    imageio.mimsave(args.save_path, frames, fps=2)

if __name__ == "__main__":
    main()

