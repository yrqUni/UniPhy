import os
import sys
import random
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

sys.path.append("/nfs/ConvLRU/Model/ConvLRU")
sys.path.append("/nfs/ConvLRU/Exp/ERA5")

from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset
from convlru_train_ddp import Args as TrainArgs
from convlru_train_ddp import load_model_args_from_ckpt, apply_model_args


class InferenceArgs:
    def __init__(self):
        self.ckpt_path = "/nfs/ConvLRU/Exp/ERA5/convlru_base/ckpt/e29_s570_l0.249103.pth"
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.output_dir = "./inference_results/1"
        self.TS = 1.0
        self.ctx_len = 8
        self.gen_len = 32
        self.fps = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def save_vis_video(preds: np.ndarray, save_dir: str, sample_idx: int, fps: int = 1) -> None:
    T, C, H, W = preds.shape
    dpi = 150
    figsize = (W / dpi, H / dpi)

    for ch in range(C):
        data_flat = preds[:, ch].flatten()
        vmin = np.percentile(data_flat, 2)
        vmax = np.percentile(data_flat, 98)

        range_span = vmax - vmin
        vmin -= range_span * 0.05
        vmax += range_span * 0.05

        save_name = f"pred_sample_{sample_idx}_ch{ch}_vis.mp4"
        save_path = os.path.join(save_dir, save_name)

        writer = imageio.get_writer(
            save_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            output_params=["-crf", "18", "-preset", "slow"],
        )

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        im = ax.imshow(
            preds[0, ch],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )

        for t in range(T):
            im.set_data(preds[t, ch])
            fig.canvas.draw()

            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]

            h_curr, w_curr = image.shape[:2]
            h_new = h_curr - (h_curr % 2)
            w_new = w_curr - (w_curr % 2)

            if h_new != h_curr or w_new != w_curr:
                image = image[:h_new, :w_new]

            writer.append_data(image)

        writer.close()
        plt.close(fig)


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tensor_sample_flat(x: torch.Tensor, max_elems: int) -> np.ndarray:
    x = x.detach()
    numel = x.numel()
    if numel == 0:
        return np.array([], dtype=np.float32)
    if numel <= max_elems:
        return x.float().reshape(-1).cpu().numpy()
    idx = torch.randint(0, numel, (max_elems,), device=x.device)
    flat = x.float().reshape(-1)
    return flat[idx].cpu().numpy()


def collect_weight_stats(
    model: torch.nn.Module,
    eps: float = 1e-6,
    global_sample_hist: int = 1_000_000,
) -> Tuple[List[Dict], Dict, np.ndarray]:
    rows: List[Dict] = []
    per_param_samples: List[np.ndarray] = []

    total_params = 0
    total_trainable = 0

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None:
                continue

            numel = int(p.numel())
            total_params += numel
            if p.requires_grad:
                total_trainable += numel

            if numel == 0:
                continue

            p_f = p.detach().float().reshape(-1)

            mean = float(p_f.mean().item())
            std = float(p_f.std(unbiased=False).item())
            minv = float(p_f.min().item())
            maxv = float(p_f.max().item())
            abs_mean = float(p_f.abs().mean().item())
            abs_max = float(p_f.abs().max().item())
            l2 = float(torch.linalg.vector_norm(p_f, ord=2).item())
            l1 = float(torch.linalg.vector_norm(p_f, ord=1).item())
            sparsity = float((p_f.abs() < eps).float().mean().item())

            rows.append(
                {
                    "name": name,
                    "shape": str(tuple(p.shape)),
                    "dtype": str(p.dtype).replace("torch.", ""),
                    "device": str(p.device),
                    "requires_grad": int(bool(p.requires_grad)),
                    "numel": numel,
                    "mean": mean,
                    "std": std,
                    "min": minv,
                    "max": maxv,
                    "abs_mean": abs_mean,
                    "abs_max": abs_max,
                    "l2_norm": l2,
                    "l1_norm": l1,
                    "sparsity(|w|<eps)": sparsity,
                }
            )

            per_param_budget = max(10_000, min(global_sample_hist // 10, 200_000))
            per_param_samples.append(_tensor_sample_flat(p.detach(), per_param_budget))

    if per_param_samples:
        all_samples = np.concatenate(per_param_samples, axis=0)
        if all_samples.size > global_sample_hist:
            sel = np.random.randint(0, all_samples.size, size=(global_sample_hist,))
            all_samples = all_samples[sel]
    else:
        all_samples = np.array([], dtype=np.float32)

    summary = {
        "total_params": int(total_params),
        "total_trainable_params": int(total_trainable),
        "global_sample_count": int(all_samples.size),
        "global_mean": float(all_samples.mean()) if all_samples.size else 0.0,
        "global_std": float(all_samples.std()) if all_samples.size else 0.0,
        "global_min": float(all_samples.min()) if all_samples.size else 0.0,
        "global_max": float(all_samples.max()) if all_samples.size else 0.0,
    }
    return rows, summary, all_samples


def save_stats_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_hist_all_weights(all_samples: np.ndarray, out_path: str, bins: int = 200) -> None:
    if all_samples.size == 0:
        return
    plt.figure(figsize=(10, 6), dpi=150)
    plt.hist(all_samples, bins=bins)
    plt.title("Histogram of sampled model weights")
    plt.xlabel("weight value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_topk_bars(rows: List[Dict], key: str, out_path: str, topk: int, title: str) -> None:
    if not rows:
        return
    rows_f = [r for r in rows if int(r.get("numel", 0)) > 0]
    rows_sorted = sorted(rows_f, key=lambda r: abs(float(r.get(key, 0.0))), reverse=True)[:topk]

    names = [r["name"] for r in rows_sorted][::-1]
    vals = [float(r[key]) for r in rows_sorted][::-1]

    height = max(6.0, 0.25 * len(names))
    plt.figure(figsize=(12, height), dpi=150)
    y = np.arange(len(names))
    plt.barh(y, vals)
    plt.yticks(y, names, fontsize=7)
    plt.xlabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _is_conv_weight(name: str, p: torch.Tensor) -> bool:
    if "weight" not in name:
        return False
    return p.dim() in (4, 5)


def visualize_conv_kernels(
    model: torch.nn.Module,
    out_dir: str,
    max_layers: int = 6,
    max_out_ch: int = 16,
) -> None:
    _safe_makedirs(out_dir)

    picked: List[Tuple[str, torch.Tensor]] = []
    for name, p in model.named_parameters():
        if _is_conv_weight(name, p):
            picked.append((name, p.detach().float().cpu()))

    if not picked:
        return

    def score(item: Tuple[str, torch.Tensor]) -> int:
        _, w = item
        out_ch = int(w.shape[0])
        if w.dim() == 4:
            k = int(w.shape[-1] * w.shape[-2])
        else:
            k = int(w.shape[-1] * w.shape[-2] * w.shape[-3])
        return out_ch * k

    picked = sorted(picked, key=score, reverse=True)[:max_layers]

    for li, (name, w) in enumerate(picked):
        out_ch = int(w.shape[0])
        n_show = min(out_ch, max_out_ch)

        if w.dim() == 5:
            w2 = w[:n_show].mean(dim=1).mean(dim=1)
        else:
            w2 = w[:n_show].mean(dim=1)

        w2_np = w2.numpy()

        cols = int(np.ceil(np.sqrt(n_show)))
        rows = int(np.ceil(n_show / cols))

        plt.figure(figsize=(cols * 2.2, rows * 2.2), dpi=150)
        for i in range(n_show):
            ax = plt.subplot(rows, cols, i + 1)
            k = w2_np[i]
            vmin = np.percentile(k, 2)
            vmax = np.percentile(k, 98)
            ax.imshow(k, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(f"out{i}", fontsize=7)

        plt.suptitle(name, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"conv_kernels_{li:02d}.png"))
        plt.close()


def analyze_and_visualize_weights(
    model: torch.nn.Module,
    out_dir: str,
    load_info: Optional[object] = None,
) -> None:
    _safe_makedirs(out_dir)

    rows, summary, all_samples = collect_weight_stats(model)

    save_stats_csv(rows, os.path.join(out_dir, "weights_stats.csv"))

    overview_path = os.path.join(out_dir, "weights_overview.txt")
    with open(overview_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        if load_info is not None and hasattr(load_info, "missing_keys") and hasattr(load_info, "unexpected_keys"):
            missing_keys = list(getattr(load_info, "missing_keys"))
            unexpected_keys = list(getattr(load_info, "unexpected_keys"))
            f.write(f"missing_keys_count: {len(missing_keys)}\n")
            f.write(f"unexpected_keys_count: {len(unexpected_keys)}\n")
            if missing_keys:
                f.write("missing_keys:\n")
                for k in missing_keys:
                    f.write(f"{k}\n")
            if unexpected_keys:
                f.write("unexpected_keys:\n")
                for k in unexpected_keys:
                    f.write(f"{k}\n")

    plot_hist_all_weights(all_samples, os.path.join(out_dir, "hist_all_weights.png"), bins=200)
    plot_topk_bars(
        rows,
        key="abs_mean",
        out_path=os.path.join(out_dir, "layer_absmean_top.png"),
        topk=40,
        title="Top layers by abs_mean(weight)",
    )
    plot_topk_bars(
        rows,
        key="l2_norm",
        out_path=os.path.join(out_dir, "layer_l2norm_top.png"),
        topk=40,
        title="Top layers by L2 norm(weight)",
    )
    visualize_conv_kernels(model, os.path.join(out_dir, "conv_kernels"), max_layers=6, max_out_ch=16)


def main() -> None:
    cfg = InferenceArgs()
    _safe_makedirs(cfg.output_dir)

    model_args = TrainArgs()
    ckpt_args_dict = load_model_args_from_ckpt(cfg.ckpt_path, map_location="cpu")
    if ckpt_args_dict:
        apply_model_args(model_args, ckpt_args_dict, verbose=False)

    model_args.data_root = cfg.data_root

    model = ConvLRU(model_args).to(cfg.device)
    model.eval()

    checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)
    state_dict = checkpoint["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    load_info = model.load_state_dict(new_state_dict, strict=False)

    analyze_and_visualize_weights(model, cfg.output_dir, load_info=load_info)

    dataset = ERA5_Dataset(
        input_dir=cfg.data_root,
        year_range=model_args.year_range,
        is_train=False,
        sample_len=cfg.ctx_len,
        eval_sample=8,
        max_cache_size=128,
    )

    sample_idx = random.randint(0, len(dataset) - 1)
    raw_sample = dataset[sample_idx]
    raw_sample = torch.from_numpy(raw_sample).unsqueeze(0).to(cfg.device).float()

    x_ctx = raw_sample

    dt_native = 1.0
    listT_ctx = torch.full((1, cfg.ctx_len), dt_native, device=cfg.device)
    listT_future = torch.full((1, cfg.gen_len), float(cfg.TS), device=cfg.device)

    static_feats = None
    static_path = os.path.join(os.path.dirname(cfg.ckpt_path), "../../static_feats.pt")
    if not os.path.exists(static_path):
        static_path = "/nfs/ConvLRU/Exp/ERA5/static_feats.pt"

    if os.path.exists(static_path) and getattr(model_args, "static_ch", 0) > 0:
        static_data = torch.load(static_path, map_location=cfg.device)
        static_feats = static_data.unsqueeze(0).repeat(1, 1, 1, 1).float()

    with torch.no_grad():
        out_gen = model(
            x_ctx,
            mode="i",
            out_gen_num=cfg.gen_len,
            listT=listT_ctx,
            listT_future=listT_future,
            static_feats=static_feats,
            timestep=None,
        )

    preds_tensor = out_gen[0]
    if preds_tensor.dim() == 5:
        preds_tensor = preds_tensor.squeeze(0)

    preds_np = preds_tensor.cpu().numpy()
    save_vis_video(preds_np, cfg.output_dir, sample_idx, fps=cfg.fps)


if __name__ == "__main__":
    main()

