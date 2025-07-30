import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/nfs/yrqUni/Workspace/ConvLRU/Model/ConvLRU')
sys.path.append('/nfs/yrqUni/Workspace/ERA5')
from ModelConvLRU_ERA5 import ConvLRU
from ERA5 import ERA5_Dataset

from convlru_train_ddp import Args

import argparse

def find_latest_ckpt(ckpt_dir):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpt_files:
        return None
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]

def get_prefix(keys):
    if len(keys) == 0:
        return ''
    key = keys[0]
    if key.startswith('module._orig_mod.'):
        return 'module._orig_mod.'
    elif key.startswith('module.'):
        return 'module.'
    else:
        return ''

def adapt_state_dict_keys(state_dict, model):
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    ckpt_prefix = get_prefix(ckpt_keys)
    model_prefix = get_prefix(model_keys)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if ckpt_prefix != '':
            new_k = k[len(ckpt_prefix):]
        if model_prefix != '':
            new_k = model_prefix + new_k
        new_state_dict[new_k] = v
    return new_state_dict

def load_model(args, ckpt_path, device):
    model = ConvLRU(args)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = adapt_state_dict_keys(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def plot_channels(input_last, pred_last, gt_last, channels, save_path):
    fig, axes = plt.subplots(len(channels), 3, figsize=(13, 3*len(channels)))
    if len(channels) == 1:
        axes = axes[None, :]
    for idx, ch in enumerate(channels):
        row = [input_last[ch], pred_last[ch], gt_last[ch]]
        vmin = min([x.min() for x in row])
        vmax = max([x.max() for x in row])
        for j, data in enumerate(row):
            im = axes[idx, j].imshow(data, aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx, j].set_title(['Input', 'Pred', 'GT'][j] + f' ch{ch}')
            axes[idx, j].axis('off')
            fig.colorbar(im, ax=axes[idx, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path)
    plt.close()

def plot_ckpt_not_found(save_path):
    plt.figure(figsize=(6,4))
    plt.text(0.5, 0.5, "Checkpoint Not Found", fontsize=24, ha='center', va='center')
    plt.axis('off')
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='', help='ckpt path (empty means use latest)')
    parser.add_argument('--ckpt_dir', type=str, default='./convlru_base/ckpt', help='ckpt directory')
    parser.add_argument('--channels', type=int, nargs='+', required=True)
    parser.add_argument('--save_prefix', type=str, default='result', help='save image prefix')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args_cli = parser.parse_args()

    model_args = Args()
    if args_cli.ckpt:
        ckpt_path = args_cli.ckpt
    else:
        ckpt_path = find_latest_ckpt(args_cli.ckpt_dir)
        if ckpt_path:
            print(f'[INFO] Auto load latest ckpt: {ckpt_path}')
    device = torch.device(args_cli.device)

    if not ckpt_path or not os.path.isfile(ckpt_path):
        for idx in range(args_cli.n):
            save_path = f'{args_cli.save_prefix}_idx{idx}_channels_{"_".join(map(str,args_cli.channels))}.png'
            plot_ckpt_not_found(save_path)
        print('Checkpoint not found, generated notice images.')
        return

    model = load_model(model_args, ckpt_path, device)

    eval_dataset = ERA5_Dataset(
        input_dir=model_args.data_root,
        year_range=model_args.year_range,
        is_train=False,
        sample_len=model_args.eval_data_n_frames,
        eval_sample=model_args.eval_sample_num,
        max_cache_size=4,
        rank=0, gpus=1
    )
    for idx in range(args_cli.n):
        data = eval_dataset[idx]
        data = torch.from_numpy(data).unsqueeze(0).to(device).float()
        inputs = data[:, :model_args.eval_data_n_frames//2]
        gts = data[:, model_args.eval_data_n_frames//2:]
        out_gen_num = gts.shape[1] // model_args.gen_factor
        with torch.no_grad():
            preds = model(inputs, 'i', out_gen_num=out_gen_num, gen_factor=model_args.gen_factor)
        input_last = inputs[0, -1].cpu().numpy()
        pred_last = preds[0, -1].cpu().numpy()
        gt_last = gts[0, -1].cpu().numpy()
        save_path = f'{args_cli.save_prefix}_idx{idx}_channels_{"_".join(map(str,args_cli.channels))}.png'
        plot_channels(input_last, pred_last, gt_last, args_cli.channels, save_path)
        del data, inputs, gts, preds
        gc.collect()
        torch.cuda.empty_cache()
    print('Done.')

if __name__ == '__main__':
    main()

