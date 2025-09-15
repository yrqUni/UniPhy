import sys
sys.path.append('/home/ruiqingyan/Workspace/ConvLRU/Model/ConvLRU')
sys.path.append('/home/ruiqingyan/Workspace/ERA5')

import gc
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset

ARGS = {
    'ckpt': 'e16_s666_l0.013081.pth',
    'data_root': '/mnt/esm10/ERA5_data/data_norm',
    'year_start': 2000,
    'year_end': 2021,
    'sample_len': 4,
    'batch_size': 1,
    'save': 'eval_viz.png',
    'channels': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

MODEL_ARG_KEYS = [
    'input_size','input_ch','use_cbam','use_gate',
    'use_freq_prior','use_sh_prior','freq_rank','freq_gain_init',
    'sh_Lmax','sh_rank','sh_gain_init','lru_rank',
    'emb_ch','convlru_num_blocks','hidden_factor', 'freq_mode',
    'emb_hidden_ch','emb_hidden_layers_num','emb_strategy',
    'ffn_hidden_ch','ffn_hidden_layers_num',
    'dec_hidden_ch','dec_hidden_layers_num','dec_strategy',
    'out_ch','gen_factor',
    'hidden_activation','output_activation',
]

def extract_model_args_from_ckpt(ckpt_path, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise RuntimeError('checkpoint must be a dict containing model args')
    if 'model_args' in ckpt and isinstance(ckpt['model_args'], dict):
        d = {k: ckpt['model_args'][k] for k in MODEL_ARG_KEYS if k in ckpt['model_args']}
    elif 'args_all' in ckpt and isinstance(ckpt['args_all'], dict):
        a = ckpt['args_all']
        d = {k: a[k] for k in MODEL_ARG_KEYS if k in a}
    else:
        raise RuntimeError('missing model_args or args_all in checkpoint')
    for k in ['model','optimizer','scheduler']:
        if k in ckpt:
            del ckpt[k]
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not d:
        raise RuntimeError('no usable model args found')
    return d

def make_args_from_overrides(overrides):
    class A: ...
    args = A()
    for k, v in overrides.items():
        setattr(args, k, v)
    return args

def adapt_state_dict_keys(state_dict, model):
    def pref(keys):
        if not keys:
            return ''
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
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        return obj
    return obj

def build_model_from_ckpt(ckpt_path, device):
    sd_raw = load_state_dict_safely(ckpt_path, device)
    ma = extract_model_args_from_ckpt(ckpt_path, map_location=device)
    if 'lru_rank' not in ma:
        for k, v in sd_raw.items():
            if k.endswith('lru_layer.U_row'):
                ma['lru_rank'] = int(v.shape[-1])
                break
    args = make_args_from_overrides(ma)
    model = ConvLRU(args).to(device)
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

def make_grid_figure(pred_btchw, gt_btchw, channels, figsize_per_cell, cmap_main, cmap_diff, save_path):
    B, T, C, H, W = pred_btchw.shape
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

def parse_channels(ch_arg, C):
    if ch_arg is None:
        return list(range(C))
    if isinstance(ch_arg, str):
        s = ch_arg.strip().lower()
        if s in ('', 'all', '*'):
            return list(range(C))
        out = []
        for x in s.split(','):
            x = x.strip()
            if not x:
                continue
            out.append(int(x))
        return sorted(set([i for i in out if 0 <= i < C]))
    if isinstance(ch_arg, (list, tuple)):
        out = [int(i) for i in ch_arg]
        return sorted(set([i for i in out if 0 <= i < C]))
    raise ValueError(f'Unsupported channels arg: {ch_arg!r}')

def main():
    args = ARGS
    device = torch.device(args['device'])
    model, margs = build_model_from_ckpt(args['ckpt'], device)
    ds = ERA5_Dataset(
        input_dir=args['data_root'],
        year_range=[args['year_start'], args['year_end']],
        is_train=False,
        sample_len=args['sample_len'],
        eval_sample=4,
        max_cache_size=32,
        rank=0,
        gpus=1
    )
    dl = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=1, pin_memory=True)
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
    C = preds.shape[2]
    ch = parse_channels(args['channels'], C)
    make_grid_figure(
        preds,
        out,
        ch,
        figsize_per_cell=(3, 3),
        cmap_main='viridis',
        cmap_diff='RdBu_r',
        save_path=args['save']
    )

if __name__ == '__main__':
    main()
