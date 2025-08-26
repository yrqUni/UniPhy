import sys
sys.path.append('/home/ruiqingyan/Workspace/ConvLRU/Model/ConvLRU')
sys.path.append('/home/ruiqingyan/Workspace/ERA5')

import argparse
import gc
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset

MODEL_ARG_KEYS = [
    'input_size','input_ch','use_mhsa','use_gate','emb_ch','convlru_num_blocks','hidden_factor',
    'emb_hidden_ch','emb_hidden_layers_num','ffn_hidden_ch','ffn_hidden_layers_num',
    'dec_hidden_ch','dec_hidden_layers_num','out_ch','gen_factor','hidden_activation','output_activation',
    'emb_strategy','dec_strategy'
]

DEFAULT_ARGS = {
    'input_size': (720, 1440),
    'input_ch': 24,
    'use_mhsa': True,
    'use_gate': True,
    'emb_ch': 48,
    'convlru_num_blocks': 8,
    'hidden_factor': (10, 20),
    'emb_strategy': 'pxus',
    'emb_hidden_ch': 1,
    'emb_hidden_layers_num': 72,
    'ffn_hidden_ch': 96,
    'ffn_hidden_layers_num': 2,
    'dec_strategy': 'pxsf',
    'dec_hidden_ch': 0,
    'dec_hidden_layers_num': 0,
    'out_ch': 24,
    'gen_factor': 1,
    'hidden_activation': 'Tanh',
    'output_activation': 'Tanh',
}

def extract_model_args_from_ckpt(ckpt_path, map_location='cpu'):
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location)
    except Exception:
        return {}
    d = {}
    if isinstance(ckpt, dict):
        if 'model_args' in ckpt and isinstance(ckpt['model_args'], dict):
            d = {k: ckpt['model_args'][k] for k in MODEL_ARG_KEYS if k in ckpt['model_args']}
        elif 'args_all' in ckpt and isinstance(ckpt['args_all'], dict):
            a = ckpt['args_all']
            d = {k: a[k] for k in MODEL_ARG_KEYS if k in a}
        for k in ['model','optimizer','scheduler']:
            if k in ckpt:
                del ckpt[k]
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return d

def make_args_from_defaults(overrides: dict):
    class A: pass
    args = A()
    for k, v in DEFAULT_ARGS.items():
        setattr(args, k, v)
    for k, v in (overrides or {}).items():
        if k in MODEL_ARG_KEYS:
            setattr(args, k, v)
    if hasattr(args, 'emb_strategy') and isinstance(args.emb_strategy, str):
        args.emb_strategy = args.emb_strategy.lower()
    if hasattr(args, 'dec_strategy') and isinstance(args.dec_strategy, str):
        args.dec_strategy = args.dec_strategy.lower()
    return args

def adapt_state_dict_keys(state_dict, model):
    def pref(keys):
        if not keys: return ''
        k = keys[0]
        if k.startswith('module._orig_mod.'):
            return 'module._orig_mod.'
        if k.startswith('module.'):
            return 'module.'
        return ''
    mk = list(model.state_dict().keys())
    ck = list(state_dict.keys())
    cpre = pref(ck)
    mpre = pref(mk)
    out = {}
    for k, v in state_dict.items():
        nk = k
        if cpre:
            nk = k[len(cpre):]
        if mpre:
            nk = mpre + nk
        out[nk] = v
    return out

def load_state_dict_safely(ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        return obj
    return obj

def build_model_from_ckpt(ckpt_path, device):
    ma = extract_model_args_from_ckpt(ckpt_path, map_location=device)
    args = make_args_from_defaults(ma)
    model = ConvLRU(args).to(device)
    sd_raw = load_state_dict_safely(ckpt_path, device)
    sd = adapt_state_dict_keys(sd_raw, model)
    model.load_state_dict(sd, strict=False)
    del sd, sd_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    return model, args

def calc_acc(pred, gt):
    f = pred.flatten().double()
    o = gt.flatten().double()
    f = f - f.mean()
    o = o - o.mean()
    num = (f * o).sum()
    den = torch.sqrt((f.pow(2)).sum()) * torch.sqrt((o.pow(2)).sum())
    den_val = den.item()
    if den_val == 0.0:
        return 0.0
    return (num / den).item()

def make_grid_figure(pred_btchw, gt_btchw, channels=None,
                     figsize_per_cell=(3, 3), cmap_main='viridis', cmap_diff='RdBu_r',
                     save_path='eval_viz.png'):
    B, T, C, H, W = pred_btchw.shape
    if channels is None:
        channels = list(range(C))
    nrows = len(channels) * 3
    ncols = T
    vmins, vmaxs = {}, {}
    for c in channels:
        v = torch.cat([pred_btchw[:, :, c].reshape(-1), gt_btchw[:, :, c].reshape(-1)], dim=0)
        vmins[c] = float(torch.min(v).item())
        vmaxs[c] = float(torch.max(v).item())
        if vmins[c] >= vmaxs[c]:
            vmaxs[c] = vmins[c] + 1e-6
    fig_w = ncols * figsize_per_cell[0]
    fig_h = nrows * figsize_per_cell[1]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    accs = {}
    for i, c in enumerate(channels):
        accs[c] = []
        for t in range(T):
            r0 = i * 3
            axes[r0][t].imshow(pred_btchw[0, t, c].cpu(), cmap=cmap_main, vmin=vmins[c], vmax=vmaxs[c])
            axes[r0][t].set_xticks([]); axes[r0][t].set_yticks([])
            if t == 0: axes[r0][t].set_ylabel(f'ch{c}-Pred')
            axes[r0+1][t].imshow(gt_btchw[0, t, c].cpu(), cmap=cmap_main, vmin=vmins[c], vmax=vmaxs[c])
            axes[r0+1][t].set_xticks([]); axes[r0+1][t].set_yticks([])
            if t == 0: axes[r0+1][t].set_ylabel(f'ch{c}-GT')
            diff = (pred_btchw[0, t, c] - gt_btchw[0, t, c]).cpu()
            rng = max(vmaxs[c] - vmins[c], 1e-6)
            dv = 0.2 * rng
            axes[r0+2][t].imshow(diff, cmap=cmap_diff, vmin=-dv, vmax=+dv)
            axes[r0+2][t].set_xticks([]); axes[r0+2][t].set_yticks([])
            if t == 0: axes[r0+2][t].set_ylabel(f'ch{c}-Diff')
            accs[c].append(calc_acc(pred_btchw[0, t, c], gt_btchw[0, t, c]))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    for c in channels:
        print(f'ACC ch{c}: {[round(x, 3) for x in accs[c]]}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--year_start', type=int, default=2000)
    p.add_argument('--year_end', type=int, default=2021)
    p.add_argument('--sample_len', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--save', type=str, default='eval_viz.png')
    p.add_argument('--channels', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    device = torch.device(args.device)
    model, margs = build_model_from_ckpt(args.ckpt, device)
    ds = ERA5_Dataset(
        input_dir=args.data_root,
        year_range=[args.year_start, args.year_end],
        is_train=False,
        sample_len=args.sample_len,
        eval_sample=4,
        max_cache_size=32,
        rank=0,
        gpus=1
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    data = next(iter(dl)).to(device).to(torch.float32)[:, :, :, 1:, :]
    T = data.shape[1]
    inp = data[:, : T // 2]
    out = data[:, T // 2 :]
    with torch.no_grad():
        out_gen_num = out.shape[1] // getattr(margs, 'gen_factor', 1)
        preds = model(inp, 'i', out_gen_num=out_gen_num, gen_factor=getattr(margs, 'gen_factor', 1))
    if preds.dim() == 4:
        preds = preds.unsqueeze(0)
    if out.dim() == 4:
        out = out.unsqueeze(0)
    if args.channels:
        ch = [int(x) for x in args.channels.split(',') if x.strip() != '']
    else:
        ch = list(range(preds.shape[2]))
    make_grid_figure(preds, out, channels=ch, save_path=args.save)

if __name__ == '__main__':
    main()
