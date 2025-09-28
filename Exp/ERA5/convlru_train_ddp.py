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
        self.emb_strategy = 'pxus'          # 'pxus' | 'conv'
        self.hidden_factor = (7, 12)
        self.emb_ch = 240
        self.emb_hidden_ch = 270
        self.emb_hidden_layers_num = 2
        self.convlru_num_blocks = 8
        self.use_cbam = True
        self.ffn_hidden_ch = 270
        self.ffn_hidden_layers_num = 2
        self.use_gate = True
        self.lru_rank = 128
        self.use_freq_prior = True
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"           # 'linear' | 'exp'
        self.use_sh_prior = True
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lambda_type = "exogenous"      # 'static' | 'exogenous'
        self.exo_mode = "mlp"               # 'mlp' | 'affine'
        self.lambda_mlp_hidden = 16
        self.dec_strategy = 'pxsf'          # 'pxsf' | 'deconv'
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0

def setup_ddp(rank, world_size, master_addr, master_port, local_rank):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    dist.destroy_process_group()

def keep_latest_ckpts(ckpt_dir):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    if len(ckpt_files) <= 3:
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    for file_path in ckpt_files[3:]:
        try:
            os.remove(file_path)
        except Exception:
            pass

MODEL_ARG_KEYS = [
    'input_size','input_ch','out_ch','hidden_activation','output_activation',
    'emb_strategy','hidden_factor','emb_ch','emb_hidden_ch','emb_hidden_layers_num',
    'convlru_num_blocks','use_cbam','use_gate','lru_rank',
    'use_freq_prior','freq_rank','freq_gain_init','freq_mode',
    'use_sh_prior','sh_Lmax','sh_rank','sh_gain_init',
    'lambda_type','exo_mode','lambda_mlp_hidden',
    'ffn_hidden_ch','ffn_hidden_layers_num',
    'dec_strategy','dec_hidden_ch','dec_hidden_layers_num'
]

def extract_model_args(args_obj):
    return {k: getattr(args_obj, k) for k in MODEL_ARG_KEYS if hasattr(args_obj, k)}

def apply_model_args(args_obj, model_args_dict, verbose=True):
    if not model_args_dict:
        return
    for k, v in model_args_dict.items():
        if hasattr(args_obj, k):
            old = getattr(args_obj, k)
            if verbose and old != v:
                msg = f"[Args] restore '{k}': {old} -> {v}"
                print(msg)
                logging.info(msg)
            setattr(args_obj, k, v)

def load_model_args_from_ckpt(ckpt_path, map_location='cpu'):
    if not os.path.isfile(ckpt_path):
        print(f"[Args] ckpt not found: {ckpt_path}")
        logging.warning(f"[Args] ckpt not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model_args = ckpt.get('model_args', None)
    if model_args is None:
        args_all = ckpt.get('args_all', None)
        if isinstance(args_all, dict):
            model_args = {k: args_all[k] for k in MODEL_ARG_KEYS if k in args_all}
    for k in ['model','optimizer','scheduler']:
        if k in ckpt:
            del ckpt[k]
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    if not model_args:
        print(f"[Args] Warning: no model_args found in ckpt, using code defaults.")
        logging.warning(f"[Args] no model_args found in ckpt, using code defaults.")
        return None
    return model_args

def save_ckpt(model, opt, epoch, step, loss, args, scheduler=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        'model': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': float(loss),
        'args_all': dict(vars(args)),
        'model_args': extract_model_args(args),
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, os.path.join(args.ckpt_dir, f'e{epoch}_s{step}_l{state["loss"]:.6f}.pth'))
    keep_latest_ckpts(args.ckpt_dir)
    del state
    gc.collect()
    torch.cuda.empty_cache()

def get_prefix(keys):
    if not keys:
        return ''
    key = keys[0]
    if key.startswith('module._orig_mod.'):
        return 'module._orig_mod.'
    if key.startswith('module.'):
        return 'module.'
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

def load_ckpt(model, opt, ckpt_path, scheduler=None, map_location='cpu', args=None, restore_model_args=False):
    if not os.path.isfile(ckpt_path):
        print(f"No checkpoint Found")
        logging.warning(f"No checkpoint Found at {ckpt_path}")
        gc.collect()
        torch.cuda.empty_cache()
        return 0, 0
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if restore_model_args and args is not None:
        model_args = checkpoint.get('model_args', None)
        if model_args is None:
            args_all = checkpoint.get('args_all', None)
            if isinstance(args_all, dict):
                model_args = {k: args_all[k] for k in MODEL_ARG_KEYS if k in args_all}
        apply_model_args(args, model_args, verbose=True)
    state_dict = adapt_state_dict_keys(checkpoint['model'], model)
    model.load_state_dict(state_dict, strict=False)
    opt.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    del state_dict
    for k in ['model','optimizer','scheduler']:
        if k in checkpoint:
            del checkpoint[k]
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
    logging.info(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
    return epoch, step

def setup_logging(args):
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

_LAT_WEIGHT_CACHE = {}

def get_latitude_weights(H, device, dtype):
    key = (H, device, dtype)
    if key in _LAT_WEIGHT_CACHE:
        return _LAT_WEIGHT_CACHE[key]
    lat_edges = torch.linspace(-90, 90, steps=H+1, device=device, dtype=dtype)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    w = torch.cos(lat_centers * torch.pi / 180.0).clamp_min(0)
    w = w / w.mean()
    _LAT_WEIGHT_CACHE[key] = w
    return w

def latitude_weighted_l1(preds, targets):
    B, T, C, H, W = preds.shape
    device = preds.device
    dtype = preds.dtype
    w = get_latitude_weights(H, device, dtype).view(1, 1, 1, H, 1)
    return ((preds - targets).abs() * w).mean()

_LRU_GATE_MEAN = {}

def register_lru_gate_hooks(ddp_model, rank):
    model_to_hook = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    for name, module in model_to_hook.named_modules():
        if name.endswith('lru_layer.gate_conv'):
            tag = None
            if 'convlru_blocks.' in name:
                try:
                    tag = int(name.split('convlru_blocks.')[1].split('.')[0])
                except:
                    tag = name
            else:
                tag = name
            def _hook(mod, inp, out, tag_local=tag):
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
        parts.append(f"g[b{k}]={v:.4f}" if isinstance(k,int) else f"g[{k}]={v:.4f}")
    return " ".join(parts)

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
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_model_args = load_model_args_from_ckpt(args.ckpt, map_location=f'cuda:{local_rank}')
        if ckpt_model_args:
            print("[Args] applying model args from ckpt before building model.")
            logging.info("[Args] applying model args from ckpt before building model.")
            apply_model_args(args, ckpt_model_args, verbose=True)
    if rank == 0:
        logging.info("==== Training Arguments (Updated) ====")
        for k, v in vars(args).items():
            logging.info(f"{k}: {v}")
        logging.info("======================================")
    model = ConvLRU(args)
    model = model.cuda(local_rank)
    loss_fn = latitude_weighted_l1 if args.loss == 'lat' else torch.nn.L1Loss()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[params] Total: {total_params:,}, Trainable: {trainable_params:,}")
        logging.info(f"[params] Total: {total_params:,}, Trainable: {trainable_params:,}")
    if args.use_compile:
        model = torch.compile(model, mode="default")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    register_lru_gate_hooks(model, rank)
    tmp_dataset = ERA5_Dataset(
        input_dir=args.data_root, year_range=args.year_range,
        is_train=True, sample_len=args.train_data_n_frames,
        eval_sample=args.eval_sample_num, max_cache_size=5,
        rank=dist.get_rank(), gpus=dist.get_world_size())
    tmp_sampler = torch.utils.data.distributed.DistributedSampler(tmp_dataset, shuffle=False)
    tmp_loader = DataLoader(tmp_dataset, sampler=tmp_sampler, batch_size=args.train_batch_size,
                            num_workers=1, pin_memory=True, prefetch_factor=1)
    len_train_dataloader = len(tmp_loader)
    del tmp_dataset, tmp_sampler, tmp_loader
    gc.collect()
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = None
    amp_dtype = torch.float16 if str(args.amp_dtype).lower() == 'fp16' else torch.bfloat16
    if args.use_amp and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 0
    scheduler = None
    if args.use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len_train_dataloader, epochs=args.epochs)
    if args.ckpt and os.path.isfile(args.ckpt):
        start_epoch, _ = load_ckpt(model, opt, args.ckpt, scheduler, map_location=f'cuda:{local_rank}', args=args, restore_model_args=False)
    if args.init_lr_scheduler and args.use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len_train_dataloader, epochs=args.epochs-start_epoch)
    if not args.use_scheduler:
        for g in opt.param_groups:
            g['lr'] = args.lr
    for ep in range(start_epoch, args.epochs):
        train_dataset = ERA5_Dataset(
            input_dir=args.data_root, year_range=args.year_range,
            is_train=True, sample_len=args.train_data_n_frames,
            eval_sample=args.eval_sample_num, max_cache_size=5,
            rank=dist.get_rank(), gpus=dist.get_world_size())
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
            num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=False)
        train_sampler.set_epoch(ep)
        train_dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{args.epochs} - Start") if rank == 0 else train_dataloader
        for train_step, data in enumerate(train_dataloader_iter, start=1):
            model.train()
            opt.zero_grad(set_to_none=True)
            data = data.cuda(local_rank, non_blocking=True).to(torch.float32)[:, :, :, :, :]
            x = data[:, :-1]
            B, L, C, H, W = x.shape
            listT = make_listT_from_arg_T(B, L, x.device, x.dtype, args.T)
            use_amp = bool(args.use_amp)
            ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            with ctx:
                preds = model(x, 'p', listT=listT)
                preds = preds[:, 1:]
                target = data[:, 2:]
                loss = loss_fn(preds, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            if args.use_scheduler and scheduler is not None:
                scheduler.step()
            loss_tensor = loss.detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (loss_tensor / world_size).item()
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0] if args.use_scheduler and scheduler is not None else opt.param_groups[0]['lr']
                gate_str = format_gate_means()
                t_str = f"T={args.T}"
                message = f"Epoch {ep+1}/{args.epochs} - Step {train_step} - Total: {avg_loss:.6f} - LR: {current_lr:.6e} - {t_str} - {gate_str}"
                if isinstance(train_dataloader_iter, tqdm):
                    train_dataloader_iter.set_description(message)
                logging.info(message)
            if rank == 0 and (train_step % max(1, int(len(train_dataloader)*args.ckpt_step)) == 0 or train_step == len(train_dataloader)):
                save_ckpt(model, opt, ep+1, train_step, avg_loss, args, scheduler if (args.use_scheduler and scheduler is not None) else None)
            del data, x, preds, target, loss, loss_tensor
            if (train_step % 16) == 0:
                gc.collect()
                torch.cuda.empty_cache()
        del train_dataset, train_sampler, train_dataloader, train_dataloader_iter
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if args.do_eval:
            model.eval()
            with torch.no_grad():
                eval_dataset = ERA5_Dataset(
                    input_dir=args.data_root, year_range=args.year_range,
                    is_train=False, sample_len=args.eval_data_n_frames,
                    eval_sample=args.eval_sample_num, max_cache_size=5,
                    rank=dist.get_rank(), gpus=dist.get_world_size())
                eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
                eval_dataloader = DataLoader(
                    eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                    num_workers=1, pin_memory=True, prefetch_factor=1)
                eval_dataloader_iter = tqdm(eval_dataloader, desc=f"Eval Epoch {ep+1}/{args.epochs}") if rank == 0 else eval_dataloader
                for eval_step, data in enumerate(eval_dataloader_iter, start=1):
                    data = data.cuda(local_rank, non_blocking=True).to(torch.float32)[:, :, :, :, :]
                    half = args.eval_data_n_frames // 2
                    cond = data[:, :half]
                    B, Lc, C, H, W = cond.shape
                    listT_cond = make_listT_from_arg_T(B, Lc, cond.device, cond.dtype, args.T)
                    out_gen_num = data.shape[1] - half
                    listT_future = make_listT_from_arg_T(B, out_gen_num, cond.device, cond.dtype, args.T)
                    with torch.cuda.amp.autocast(enabled=bool(args.use_amp), dtype=amp_dtype):
                        preds = model(cond, mode="i", out_gen_num=out_gen_num, listT=listT_cond, listT_future=listT_future)
                        loss_eval = loss_fn(preds, data[:, half:])
                    tot_tensor = loss_eval.detach()
                    dist.all_reduce(tot_tensor, op=dist.ReduceOp.SUM)
                    avg_total = (tot_tensor / world_size).item()
                    if rank == 0:
                        message = f"Eval step {eval_step} - Total: {avg_total:.6f}"
                        if isinstance(eval_dataloader_iter, tqdm):
                            eval_dataloader_iter.set_description(message)
                        logging.info(message)
                    del data, cond, preds, loss_eval, tot_tensor
                    gc.collect(); torch.cuda.empty_cache()
                del eval_dataset, eval_sampler, eval_dataloader, eval_dataloader_iter
                gc.collect()
                torch.cuda.empty_cache()
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
