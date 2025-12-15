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
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

sys.path.append('/nfs/ConvLRU/Model/ConvLRU')
sys.path.append('/nfs/ConvLRU/Exp/ERA5')

from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset

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

class Args:
    def __init__(self):
        self.input_size = (721, 1440)
        self.input_ch = 30
        self.out_ch = 30
        self.hidden_activation = 'Tanh'
        self.output_activation = 'Tanh'
        self.emb_strategy = 'pxus'
        self.hidden_factor = (7, 12)
        self.emb_ch = 90
        self.emb_hidden_ch = 120
        self.emb_hidden_layers_num = 2
        self.convlru_num_blocks = 6
        self.use_cbam = True
        self.ffn_hidden_ch = 120
        self.ffn_hidden_layers_num = 2
        self.use_gate = True
        self.lru_rank = 32
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
        self.epochs = 128
        self.log_path = './convlru_base/logs'
        self.ckpt_dir = './convlru_base/ckpt'
        self.ckpt_step = 0.25
        self.do_eval = False
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-5
        self.use_scheduler = False
        self.init_lr_scheduler = False
        self.loss = 'nll'
        self.T = 6
        self.use_amp = False
        self.amp_dtype = 'fp16'
        self.grad_clip = 0.30
        self.sample_k = 9
        self.use_wandb = True
        self.wandb_project = "ERA5"
        self.wandb_entity = "ConvLRU"
        self.wandb_run_name = self.ckpt
        self.wandb_group = "v2.1.0"
        self.wandb_mode = "online"

def setup_ddp(rank, world_size, master_addr, master_port, local_rank):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=1800))
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def keep_latest_ckpts(ckpt_dir):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    if len(ckpt_files) <= 64:
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    for file_path in ckpt_files[64:]:
        try:
            os.remove(file_path)
        except Exception:
            pass

def setup_logging(args):
    if not dist.is_initialized() or dist.get_rank() != 0:
        return
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_LAT_WEIGHT_CACHE = {}

def get_latitude_weights(H, device, dtype):
    key = (H, device, dtype)
    if key in _LAT_WEIGHT_CACHE:
        return _LAT_WEIGHT_CACHE[key]
    lat_edges = torch.linspace(-90, 90, steps=H + 1, device=device, dtype=dtype)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    w = torch.cos(lat_centers * torch.pi / 180.0).clamp_min(0)
    w = w / w.mean()
    _LAT_WEIGHT_CACHE[key] = w
    return w

def gaussian_nll_loss_weighted(preds, targets):
    B, L, C2, H, W = preds.shape
    C = C2 // 2
    mu = preds[:, :, :C]
    sigma = preds[:, :, C:]
    device = preds.device
    dtype = preds.dtype
    w = get_latitude_weights(H, device, dtype).view(1, 1, 1, H, 1)
    var = sigma.pow(2)
    nll = 0.5 * (torch.log(var) + (targets - mu).pow(2) / var)
    weighted_nll = (nll * w).mean()
    return weighted_nll

def latitude_weighted_l1(preds, targets):
    B, T, C_pred, H, W = preds.shape
    B, T, C_gt, H, W = targets.shape
    device = preds.device
    dtype = preds.dtype
    if C_pred == 2 * C_gt:
        preds = preds[:, :, :C_gt]
    w = get_latitude_weights(H, device, dtype).view(1, 1, 1, H, 1)
    return ((preds - targets).abs() * w).mean()

_LRU_GATE_MEAN = {}

def register_lru_gate_hooks(ddp_model, rank):
    model_to_hook = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    for name, module in model_to_hook.named_modules():
        if name.endswith('lru_layer.gate_conv'):
            if 'convlru_blocks.' in name:
                try:
                    tag = int(name.split('convlru_blocks.')[1].split('.')[0])
                except Exception:
                    tag = name
            else:
                tag = name
            def _hook(mod, inp, out, tag_local=name):
                with torch.no_grad():
                    m = out.mean().detach()
                    _LRU_GATE_MEAN[tag_local] = float(m)
            module.register_forward_hook(_hook)

def format_gate_means():
    if not _LRU_GATE_MEAN:
        return "g=NA"
    keys = sorted(_LRU_GATE_MEAN.keys(), key=lambda k: (0, k) if isinstance(k, int) else (1, str(k)))
    parts = []
    for k in keys:
        v = _LRU_GATE_MEAN[k]
        parts.append(f"g[b{k}]={v:.4f}" if isinstance(k, int) else f"g[{k}]={v:.4f}")
    return " ".join(parts)

def get_grad_stats(model):
    total_norm_sq = 0.0
    max_abs = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_count += 1
        param_norm = p.grad.data.norm(2)
        total_norm_sq += param_norm.item() ** 2
        max_abs = max(max_abs, p.grad.data.abs().max().item())
    total_norm = total_norm_sq ** 0.5 if param_count > 0 else 0.0
    return float(total_norm), float(max_abs), param_count

def make_listT_from_arg_T(B, L, device, dtype, T):
    if T is None or T < 0:
        return None
    return torch.full((B, L), float(T), device=device, dtype=dtype)

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

def setup_wandb(rank, args, model):
    if rank != 0 or not getattr(args, "use_wandb", False):
        return
    wandb_kwargs = {
        "project": args.wandb_project,
        "config": vars(args),
    }
    if args.wandb_entity is not None:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        wandb_kwargs["name"] = args.wandb_run_name
    if args.wandb_group is not None:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_mode is not None:
        wandb_kwargs["mode"] = args.wandb_mode
    wandb.init(**wandb_kwargs)
    if isinstance(model, DDP):
        wandb.watch(model.module, log="all", log_freq=100)
    else:
        wandb.watch(model, log="all", log_freq=100)

MODEL_ARG_KEYS = [
    'input_size', 'input_ch', 'out_ch', 'hidden_activation', 
    'emb_strategy', 'hidden_factor', 'emb_ch', 'emb_hidden_ch', 'emb_hidden_layers_num',
    'convlru_num_blocks', 'use_cbam', 'use_gate', 'lru_rank',
    'use_freq_prior', 'freq_rank', 'use_sh_prior', 'sh_Lmax',
    'ffn_hidden_ch', 'ffn_hidden_layers_num',
    'dec_strategy', 'dec_hidden_ch'
]

def extract_model_args(args_obj):
    return {k: getattr(args_obj, k) for k in MODEL_ARG_KEYS if hasattr(args_obj, k)}

def apply_model_args(args_obj, model_args_dict, verbose=True):
    if not model_args_dict: return
    for k, v in model_args_dict.items():
        if hasattr(args_obj, k):
            old = getattr(args_obj, k)
            if verbose and old != v:
                msg = f"[Args] restore '{k}': {old} -> {v}"
                print(msg)
                if dist.get_rank() == 0: logging.info(msg)
            setattr(args_obj, k, v)

def load_ckpt(model, opt, ckpt_path, scheduler=None, map_location='cpu', args=None, restore_model_args=False):
    if not os.path.isfile(ckpt_path):
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if restore_model_args and args is not None:
        model_args = checkpoint.get('model_args', None)
        if model_args: apply_model_args(args, model_args)
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not isinstance(model, DDP):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    opt.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    print(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
    return epoch, step

def save_ckpt(model, opt, epoch, step, loss, args, scheduler=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        'model': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': float(loss),
        'args_all': dict(vars(args)),
        'model_args': extract_model_args(args),
    }
    if scheduler: state['scheduler'] = scheduler.state_dict()
    ckpt_path = os.path.join(args.ckpt_dir, f'e{epoch}_s{step}_l{loss:.6f}.pth')
    torch.save(state, ckpt_path)
    keep_latest_ckpts(args.ckpt_dir)

def run_ddp(rank, world_size, local_rank, master_addr, master_port, args):
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if rank == 0: setup_logging(args)
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    model = ConvLRU(args).cuda(local_rank)
    if args.loss == 'nll':
        loss_fn = gaussian_nll_loss_weighted
        if rank == 0: logging.info("Using Gaussian NLL Loss (Probabilistic Training)")
    elif args.loss == 'lat':
        loss_fn = latitude_weighted_l1
    else:
        loss_fn = torch.nn.L1Loss()
    if args.use_compile:
        model = torch.compile(model, mode="default")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    register_lru_gate_hooks(model, rank)
    setup_wandb(rank, args, model)
    train_dataset = ERA5_Dataset(input_dir=args.data_root, year_range=args.year_range, is_train=True, sample_len=args.train_data_n_frames, max_cache_size=8, rank=rank, gpus=world_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1, pin_memory=True)
    len_loader = len(train_dataloader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    start_epoch = 0
    if args.ckpt:
        start_epoch, _ = load_ckpt(model, opt, args.ckpt, map_location=f'cuda:{local_rank}', args=args)
    for ep in range(start_epoch, args.epochs):
        train_sampler.set_epoch(ep)
        loader_iter = tqdm(train_dataloader, desc=f"Epoch {ep+1}") if rank == 0 else train_dataloader
        for step, data in enumerate(loader_iter, start=1):
            model.train()
            opt.zero_grad(set_to_none=True)
            B_full, L_full, C, H, W = data.shape
            L_eff = L_full - 1
            if args.sample_k > 0 and args.sample_k <= L_eff:
                idxs = make_random_indices(L_eff, args.sample_k)
                x = data[:, idxs].cuda(local_rank, non_blocking=True).float()
                listT_vals = build_dt_from_indices(idxs, args.T)
                tgt_idxs = np.clip(idxs[1:] + 1, 1, L_full - 1)
                target = data[:, tgt_idxs].cuda(local_rank, non_blocking=True).float()
            else:
                x = data[:, :L_eff].cuda(local_rank, non_blocking=True).float()
                target = data[:, 1:].cuda(local_rank, non_blocking=True).float()
                listT_vals = [float(args.T)] * x.size(1)
            listT = torch.tensor(listT_vals, device=x.device).view(1, -1).repeat(x.size(0), 1)
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=torch.float16 if args.amp_dtype=='fp16' else torch.bfloat16):
                preds = model(x, mode='p', listT=listT)
                preds = preds[:, 1:]
                loss = loss_fn(preds, target)
                
                # [FIX]: Monitor L1 loss (Mean Absolute Error) explicitly
                # Detach preds to ensure no gradient leak
                with torch.no_grad():
                    metric_l1 = latitude_weighted_l1(preds.detach(), target)

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if rank == 0:
                gnorm, gmax, _ = get_grad_stats(model)
            scaler.step(opt)
            scaler.update()
            
            # Aggregate NLL Loss
            loss_val = loss.detach()
            dist.all_reduce(loss_val)
            avg_loss = (loss_val / world_size).item()
            
            # [FIX]: Aggregate L1 Loss for Monitoring
            l1_val = metric_l1.detach()
            dist.all_reduce(l1_val)
            avg_l1 = (l1_val / world_size).item()
            
            if rank == 0:
                global_step = ep * len_loader + step
                wandb_dict = {
                    "train/loss_nll": avg_loss,
                    "train/loss_l1": avg_l1,
                    "train/epoch": ep + 1,
                    "train/grad_norm": gnorm,
                    "train/grad_max": gmax,
                    "train/lr": opt.param_groups[0]['lr']
                }
                if hasattr(model.module.convlru_model.convlru_blocks[0].lru_layer, 'forcing_scale'):
                    fs = model.module.convlru_model.convlru_blocks[0].lru_layer.forcing_scale.item()
                    nl = model.module.convlru_model.convlru_blocks[0].lru_layer.noise_level.item()
                    wandb_dict["phys/forcing_scale"] = fs
                    wandb_dict["phys/noise_level"] = nl
                if args.use_wandb: wandb.log(wandb_dict, step=global_step)
                loader_iter.set_description(f"Ep {ep+1} NLL {avg_loss:.4f} L1 {avg_l1:.4f}")
            
            if rank == 0 and step % int(len_loader * args.ckpt_step) == 0:
                save_ckpt(model, opt, ep+1, step, avg_loss, args)
        if args.do_eval:
            if rank == 0:
                eval_dataset = ERA5_Dataset(input_dir=args.data_root, year_range=args.year_range, is_train=False, sample_len=args.eval_data_n_frames, eval_sample=args.eval_sample_num, max_cache_size=8, rank=dist.get_rank(), gpus=dist.get_world_size())
                eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=True)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1, pin_memory=True, prefetch_factor=1)
                eval_dataloader_iter = tqdm(eval_dataloader, desc=f"Eval Epoch {ep + 1}/{args.epochs}")
                model.eval()
                with torch.no_grad():
                    for eval_step, data in enumerate(eval_dataloader_iter, start=1):
                        B_full, L_full, C, H, W = data.shape
                        half = args.eval_data_n_frames // 2
                        cond = data[:, :half].cuda(local_rank, non_blocking=True).to(torch.float32)
                        K_eval = args.sample_k if (args.sample_k != -1 and args.sample_k <= cond.shape[1]) else -1
                        if K_eval == -1:
                            cond_eff = cond
                            Bc, Lc, _, _, _ = cond_eff.shape
                            listT_cond_vals = [float(args.T)] * Lc
                        else:
                            Lc_eff = cond.shape[1]
                            idxs_c = make_random_indices(Lc_eff, K_eval)
                            cond_eff = cond[:, idxs_c]
                            listT_cond_vals = build_dt_from_indices(idxs_c, args.T)
                        listT_cond = torch.tensor(listT_cond_vals, device=cond_eff.device).view(1, -1).repeat(cond_eff.size(0), 1)
                        out_gen_num = data.shape[1] - cond_eff.shape[1]
                        listT_future = make_listT_from_arg_T(B_full, out_gen_num, cond_eff.device, cond_eff.dtype, args.T)
                        target = data[:, cond_eff.shape[1]:cond_eff.shape[1] + out_gen_num].cuda(local_rank, non_blocking=True).float()
                        preds = model(cond_eff, mode="i", out_gen_num=out_gen_num, listT=listT_cond, listT_future=listT_future)
                        loss_eval = latitude_weighted_l1(preds, target)
                        if args.use_wandb:
                            wandb.log({"eval/l1_loss": loss_eval.item()}, step=global_step)
    if rank == 0 and args.use_wandb:
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    if args.use_amp:
        print("[Warning] AMP forced off for stability.")
        args.use_amp = False
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)
