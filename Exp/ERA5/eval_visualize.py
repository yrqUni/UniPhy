import argparse
import os
import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset

MODEL_ARG_KEYS = [
    'input_size','input_ch','use_mhsa','use_gate','emb_ch','convlru_num_blocks','hidden_factor',
    'emb_hidden_ch','emb_hidden_layers_num','ffn_hidden_ch','ffn_hidden_layers_num',
    'dec_hidden_ch','dec_hidden_layers_num','out_ch','gen_factor','hidden_activation','output_activation'
]

def extract_model_args_from_ckpt(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    d = ckpt.get('model_args', None)
    if d is None:
        a = ckpt.get('args_all', None)
        if isinstance(a, dict):
            d = {k: a[k] for k in MODEL_ARG_KEYS if k in a}
    for k in ['model','optimizer','scheduler']:
        if isinstance(ckpt, dict) and k in ckpt:
            del ckpt[k]
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return d

def adapt_state_dict_keys(state_dict, model):
    def pref(keys):
        if not keys: return ''
        k = keys[0]
        if k.startswith('module._orig_mod.'): return 'module._orig_mod.'
        if k.startswith('module.'): return 'module.'
        return ''
    mk = list(model.state_dict().keys())
    ck = list(state_dict.keys())
    cpre = pref(ck); mpre = pref(mk)
    out = {}
    for k,v in state_dict.items():
        nk = k
        if cpre: nk = k[len(cpre):]
        if mpre: nk = mpre + nk
        out[nk] = v
    return out

def build_model_from_ckpt(ckpt_path, device):
    ma = extract_model_args_from_ckpt(ckpt_path, map_location=device)
    class A: pass
    args = A()
    if ma:
        for k,v in ma.items(): setattr(args,k,v)
    else:
        setattr(args,'input_size',(720,1440))
        setattr(args,'input_ch',24)
        setattr(args,'use_mhsa',True)
        setattr(args,'use_gate',True)
        setattr(args,'emb_ch',48)
        setattr(args,'convlru_num_blocks',8)
        setattr(args,'hidden_factor',(10,20))
        setattr(args,'emb_hidden_ch',1)
        setattr(args,'emb_hidden_layers_num',72)
        setattr(args,'ffn_hidden_ch',96)
        setattr(args,'ffn_hidden_layers_num',2)
        setattr(args,'dec_hidden_ch',0)
        setattr(args,'dec_hidden_layers_num',0)
        setattr(args,'out_ch',24)
        setattr(args,'gen_factor',1)
        setattr(args,'hidden_activation','Tanh')
        setattr(args,'output_activation','Tanh')
    model = ConvLRU(args).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    sd = adapt_state_dict_keys(ck['model'], model)
    model.load_state_dict(sd, strict=False)
    del sd
    for k in ['model','optimizer','scheduler']:
        if k in ck: del ck[k]
    del ck
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    return model, args

def tensor_to_uint8(x):
    x = x.detach().float().cpu().numpy()
    x = (np.clip((x+1.0)*0.5, 0.0, 1.0)*255.0).astype(np.uint8)
    return x

def make_mosaic(pred_btchw, gt_btchw, channels=None, pad=4, group_pad=8):
    B,T,C,H,W = pred_btchw.shape
    assert B==1
    if channels is None:
        channels = list(range(C))
    Cn = len(channels)
    row_h = H
    col_w = W
    h_sep = 1
    v_sep = 1
    unit_rows = 2
    total_rows = T*unit_rows + (T-1)
    total_cols = Cn + (Cn-1)
    H_img = total_rows*row_h + (T-1)*group_pad + (total_rows-1)*h_sep
    W_img = total_cols*col_w + (Cn-1)*v_sep
    canvas = np.zeros((H_img, W_img), dtype=np.uint8)
    y = 0
    for t in range(T):
        for r in range(2):
            row_src = pred_btchw if r==0 else gt_btchw
            y_row = y + r*(row_h + h_sep)
            x = 0
            for ci, c in enumerate(channels):
                img = tensor_to_uint8(row_src[0,t,c])
                canvas[y_row:y_row+row_h, x:x+col_w] = img
                x += col_w
                if ci < Cn-1:
                    canvas[y_row:y_row+row_h, x:x+v_sep] = 255
                    x += v_sep
        y += 2*row_h + h_sep
        if t < T-1:
            canvas[y:y+h_sep, :] = 255
            y += h_sep
            canvas[y:y+group_pad, :] = 255
            y += group_pad
    return canvas

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
        eval_sample=1,
        max_cache_size=2,
        rank=0,
        gpus=1
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    data = next(iter(dl)).to(device).to(torch.float32)[:, :, :, 1:, :]
    T = data.shape[1]
    inp = data[:, :T//2]
    out = data[:, T//2:]
    with torch.no_grad():
        out_gen_num = out.shape[1]//getattr(margs, 'gen_factor', 1)
        preds = model(inp, 'i', out_gen_num=out_gen_num, gen_factor=getattr(margs, 'gen_factor', 1))
    if preds.dim()==4:
        preds = preds.unsqueeze(0)
    if out.dim()==4:
        out = out.unsqueeze(0)
    if args.channels:
        ch = [int(x) for x in args.channels.split(',') if x.strip()!='']
    else:
        ch = list(range(preds.shape[2]))
    img = make_mosaic(preds, out, channels=ch)
    from PIL import Image
    Image.fromarray(img).save(args.save)

if __name__ == '__main__':
    main()

