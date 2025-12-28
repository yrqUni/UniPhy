import os
import sys
import random
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
        self.ckpt_path = "/nfs/ConvLRU/Exp/ERA5/convlru_base/ckpt/e7_s570_l0.265707.pth"
        self.data_root = "/nfs/ERA5_data/data_norm"
        self.output_dir = "./inference_results"
        self.TS = 6.0
        self.ctx_len = 8
        self.gen_len = 40
        self.fps = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def save_vis_video(preds, save_dir, sample_idx, fps=1):
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
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            output_params=['-crf', '18', '-preset', 'slow']
        )
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        
        im = ax.imshow(
            preds[0, ch], 
            cmap='magma', 
            vmin=vmin, 
            vmax=vmax, 
            interpolation='bicubic'
        )
        
        for t in range(T):
            im.set_data(preds[t, ch])
            fig.canvas.draw()
            
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
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


def main():
    cfg = InferenceArgs()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    model_args = TrainArgs()
    ckpt_args_dict = load_model_args_from_ckpt(cfg.ckpt_path, map_location="cpu")
    if ckpt_args_dict:
        apply_model_args(model_args, ckpt_args_dict, verbose=False)
    
    model_args.data_root = cfg.data_root
    
    model = ConvLRU(model_args).to(cfg.device)
    model.eval()
    
    checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)
    state_dict = checkpoint['model']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    dataset = ERA5_Dataset(
        input_dir=cfg.data_root,
        year_range=model_args.year_range,
        is_train=False,
        sample_len=cfg.ctx_len,
        eval_sample=8,
        max_cache_size=128
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
        
    if os.path.exists(static_path) and model_args.static_ch > 0:
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
            timestep=None
        )
    
    preds_tensor = out_gen[0]
    
    if preds_tensor.dim() == 5:
        preds_tensor = preds_tensor.squeeze(0)
    
    preds_np = preds_tensor.cpu().numpy()
    
    save_vis_video(preds_np, cfg.output_dir, sample_idx, fps=cfg.fps)


if __name__ == "__main__":
    main()

