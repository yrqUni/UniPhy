import random
import numpy as np
import torch
import sys
import os
import gc
import glob
import logging
import datetime
import warnings

warnings.filterwarnings("ignore")

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

set_random_seed(1017, deterministic=False)

sys.path.append('/nfs/ConvLRU/Model/ConvLRU')
sys.path.append('/nfs/ConvLRU/Exp/ERA5')

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset
from tqdm import tqdm

class Args:
    def __init__(self):
        self.input_size = (721, 1440)
        self.input_ch = 30
        self.out_ch = 30
        self.hidden_activation = 'Tanh'
        self.output_activation = 'Tanh'
        self.emb_strategy = 'pxus'
        self.hidden_factor = (7, 12)
        self.emb_ch = 180
        self.emb_hidden_ch = 210
        self.emb_hidden_layers_num = 2
        self.convlru_num_blocks = 10
        self.use_cbam = True
        self.ffn_hidden_ch = 210
        self.ffn_hidden_layers_num = 2
        self.use_gate = True
        self.lru_rank = 128
        self.use_freq_prior = True
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = True
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lambda_type = "exogenous"
        self.exo_mode = "mlp"
        self.lambda_mlp_hidden = 16
        self.dec_strategy = 'pxsf'
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.data_root = '/nfs/ERA5_data/data_norm'
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 27
        self.eval_data_n_frames = 4
        self.eval_sample_num = 1
        self.ckpt = ''
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.epochs = 1000
        self.log_path = './convlru_base/logs'
        self.ckpt_dir = './convlru_base/ckpt'
        self.ckpt_step = 0.25
        self.do_eval = False
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-4
        self.use_scheduler = False
        self.init_lr_scheduler = False
        self.loss = 'l1'
        self.T = 6
        self.use_amp = False
        self.amp_dtype = 'fp16'
        self.grad_clip = 0.0
        self.sample_k = 9

def setup_ddp(rank, world_size, master_addr, master_port, local_rank):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def setup_logging(args):
    if dist.get_rank() != 0:
        return
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_latitude_weights(H, device, dtype):
    lat_edges = torch.linspace(-90, 90, steps=H + 1, device=device, dtype=dtype)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    w = torch.cos(lat_centers * torch.pi / 180.0).clamp_min(0)
    w = w / w.mean()
    return w

def latitude_weighted_l1(preds, targets):
    B, T, C, H, W = preds.shape
    w = get_latitude_weights(H, preds.device, preds.dtype).view(1, 1, 1, H, 1)
    return ((preds - targets).abs() * w).mean()

_LRU_GATE_MEAN = {}

def register_lru_gate_hooks(ddp_model):
    model_to_hook = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    for name, module in model_to_hook.named_modules():
        if name.endswith('lru_layer.gate_conv'):
            def _hook(mod, inp, out, tag=name):
                with torch.no_grad():
                    _LRU_GATE_MEAN[tag] = float(out.mean())
            module.register_forward_hook(_hook)

def format_gate_means():
    if not _LRU_GATE_MEAN:
        return "g=NA"
    return " ".join([f"g[{k}]={v:.4f}" for k, v in _LRU_GATE_MEAN.items()])

def make_random_indices(L_eff, K):
    if K <= 0:
        return np.array([], dtype=int)
    if K == 1:
        return np.array([0], dtype=int)
    idx = np.random.choice(L_eff, size=K, replace=False)
    idx.sort()
    return idx

def build_dt_from_indices(idxs, base_T):
    if len(idxs) == 0:
        return []
    dt = [float(base_T)]
    for i in range(1, len(idxs)):
        gap = int(idxs[i] - idxs[i - 1])
        dt.append(float(base_T) * max(1, gap))
    return dt

def make_listT_from_arg_T(B, L, device, dtype, T):
    if T is None or T < 0:
        return None
    return torch.full((B, L), float(T), device=device, dtype=dtype)

def run_ddp(rank, world_size, local_rank, master_addr, master_port, args):
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0:
        setup_logging(args)
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    model = ConvLRU(args).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    register_lru_gate_hooks(model)
    loss_fn = latitude_weighted_l1 if args.loss == 'lat' else torch.nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = None
    amp_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    for ep in range(args.epochs):
        dataset = ERA5_Dataset(input_dir=args.data_root, year_range=args.year_range, is_train=True,
                               sample_len=args.train_data_n_frames, eval_sample=args.eval_sample_num,
                               max_cache_size=8, rank=rank, gpus=world_size)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, drop_last=True)
        loader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=1, pin_memory=True)
        sampler.set_epoch(ep)
        loader_iter = tqdm(loader, desc=f"Epoch {ep+1}/{args.epochs}") if rank == 0 else loader
        for step, data in enumerate(loader_iter, start=1):
            model.train()
            opt.zero_grad(set_to_none=True)
            B_full, L_full, C, H, W = data.shape
            L_eff = L_full - 1
            if args.sample_k == -1:
                x = data[:, :L_eff].cuda(local_rank)
                target = data[:, 2:L_eff+1].cuda(local_rank)
                listT_vals = [float(args.T)] * x.size(1)
                K = -1
            else:
                K = args.sample_k
                if K > L_eff:
                    x = data[:, :L_eff].cuda(local_rank)
                    target = data[:, 2:L_eff+1].cuda(local_rank)
                    listT_vals = [float(args.T)] * x.size(1)
                    K = -1
                else:
                    idxs = make_random_indices(L_eff, K)
                    x = data[:, idxs].cuda(local_rank)
                    listT_vals = build_dt_from_indices(idxs, args.T)
                    tgt_idxs = np.clip(idxs[1:] + 1, 1, L_full - 1)
                    target = data[:, tgt_idxs].cuda(local_rank)
            listT = torch.tensor(listT_vals, device=x.device, dtype=x.dtype).view(1, -1).repeat(x.size(0), 1)
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
                preds = model(x, 'p', listT=listT)
                preds = preds[:, 1:]
                loss = loss_fn(preds, target)
            loss.backward()
            opt.step()
            loss_tensor = loss.detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()
            if rank == 0:
                gate_str = format_gate_means()
                if K == -1:
                    t_str = f"T={args.T} (fixed)"
                else:
                    t_min = min(listT_vals)
                    t_max = max(listT_vals)
                    t_mean = sum(listT_vals) / len(listT_vals)
                    t_str = f"T[min/mean/max]={t_min:.2f}/{t_mean:.2f}/{t_max:.2f} (K={K})"
                msg = f"Epoch {ep+1}/{args.epochs} Step {step} Loss {avg_loss:.6f} {t_str} {gate_str}"
                loader_iter.set_description(msg)
                logging.info(msg)
            del x, target, preds, loss
            gc.collect(); torch.cuda.empty_cache()
        del dataset, sampler, loader, loader_iter
        gc.collect(); torch.cuda.empty_cache()
        dist.barrier()
    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    if bool(args.use_amp):
        print("[Warning] AMP is disabled by policy. Forcing use_amp=False.")
        logging.warning("AMP is disabled by policy. Forcing use_amp=False.")
        args.use_amp = False
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)
