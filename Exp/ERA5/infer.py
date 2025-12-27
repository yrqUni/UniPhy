import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from typing import Optional, Dict, Any

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
        self.fps = 4
        
        if torch.cuda.device_count() >= 2:
            self.dev_in = "cuda:0"
            self.dev_core = "cuda:1"
        else:
            self.dev_in = "cuda:0"
            self.dev_core = "cuda:0"

def save_all_channels_mp4(preds_list, output_dir, sample_idx, ts, fps=4):
    if isinstance(preds_list, list):
        preds_np = torch.cat(preds_list, dim=0).numpy()
    else:
        preds_np = preds_list.cpu().numpy()
    
    T, C, H, W = preds_np.shape
    
    for c in range(C):
        vmin = preds_np[:, c].min()
        vmax = preds_np[:, c].max()
        
        save_name = f"pred_sample_{sample_idx}_TS{int(ts)}_var{c}.mp4"
        save_path = os.path.join(output_dir, save_name)
        
        writer = imageio.get_writer(save_path, fps=fps, format='FFMPEG')
        
        for t in range(T):
            fig = plt.figure(figsize=(8, 8 * H / W), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            ax.imshow(preds_np[t, c], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]
            
            writer.append_data(image)
            plt.close(fig)
            
        writer.close()
        print(f"Saved {save_path}")

def main():
    cfg = InferenceArgs()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    model_args = TrainArgs()
    ckpt_args_dict = load_model_args_from_ckpt(cfg.ckpt_path, map_location="cpu")
    if ckpt_args_dict:
        apply_model_args(model_args, ckpt_args_dict, verbose=False)
    model_args.data_root = cfg.data_root
    
    model = ConvLRU(model_args)
    
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    state_dict = checkpoint['model']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    model.revin.to(cfg.dev_in)
    model.embedding.to(cfg.dev_in)
    model.decoder.to(cfg.dev_in)
    model.convlru_model.to(cfg.dev_core)
    
    model.eval()

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
    x_ctx = torch.from_numpy(raw_sample).unsqueeze(0).float()
    
    dt_native = 1.0
    listT_ctx = torch.full((1, cfg.ctx_len), dt_native)
    
    static_feats = None
    static_path = os.path.join(os.path.dirname(cfg.ckpt_path), "../../static_feats.pt")
    if not os.path.exists(static_path):
        static_path = "/nfs/ConvLRU/Exp/ERA5/static_feats.pt"
    if os.path.exists(static_path) and model_args.static_ch > 0:
        static_data = torch.load(static_path, map_location="cpu")
        static_feats = static_data.unsqueeze(0).repeat(1, 1, 1, 1).float()

    out_frames = []
    last_hidden_outs = None
    
    with torch.no_grad():
        x_ctx_in = x_ctx.to(cfg.dev_in)
        s_in = static_feats.to(cfg.dev_in) if static_feats is not None else None
        
        x_norm = model.revin(x_ctx_in, "norm")
        x_emb, cond_emb = model.embedding(x_norm, static_feats=s_in)
        
        x_emb_core = x_emb.to(cfg.dev_core)
        cond_core = cond_emb.to(cfg.dev_core) if cond_emb is not None else None
        listT_ctx_core = listT_ctx.to(cfg.dev_core)
        
        _, last_hidden_outs = model.convlru_model(
            x_emb_core, 
            last_hidden_ins=None, 
            listT=listT_ctx_core, 
            cond=cond_core
        )
        
        current_x = x_ctx_in[:, -1:]
        
        for i in range(cfg.gen_len):
            dt_step = torch.full((1, 1), float(cfg.TS), device=cfg.dev_core)
            
            x_norm_step = model.revin(current_x, "norm")
            x_emb_step, cond_step = model.embedding(x_norm_step, static_feats=s_in)
            
            x_emb_step_core = x_emb_step.to(cfg.dev_core)
            
            x_hid_step_core, last_hidden_outs = model.convlru_model(
                x_emb_step_core,
                last_hidden_ins=last_hidden_outs,
                listT=dt_step,
                cond=cond_core
            )
            
            x_hid_step_out = x_hid_step_core.to(cfg.dev_in)
            cond_out = cond_emb.to(cfg.dev_in) if cond_emb is not None else None
            
            out_dec = model.decoder(x_hid_step_out, cond=cond_out, timestep=None)
            
            out_dec = out_dec.permute(0, 2, 1, 3, 4).contiguous()
            
            if model.decoder.head_mode == "gaussian":
                mu, sigma = torch.chunk(out_dec, 2, dim=2)
                if mu.size(2) == model.revin.num_features:
                    pred_frame = model.revin(mu, "denorm")
                else:
                    pred_frame = mu
            else:
                if out_dec.size(2) == model.revin.num_features:
                    pred_frame = model.revin(out_dec, "denorm")
                else:
                    pred_frame = out_dec
            
            out_frames.append(pred_frame.cpu())
            
            current_x = pred_frame
            
            print(f"Generated frame {i+1}/{cfg.gen_len}")

    save_all_channels_mp4(out_frames, cfg.output_dir, sample_idx, cfg.TS, fps=cfg.fps)

if __name__ == "__main__":
    main()

