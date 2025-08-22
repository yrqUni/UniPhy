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
import torch.nn.functional as F
warnings.filterwarnings("ignore")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(1017)

sys.path.append('/nfs/yrqUni/Workspace/ConvLRU/Model/ConvLRU')
sys.path.append('/nfs/yrqUni/Workspace/ERA5')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset
from tqdm import tqdm

class Args:
    def __init__(self):
        self.input_size = (720, 1440)
        self.input_ch = 24
        self.use_mhsa = True
        self.use_gate = True
        self.emb_ch = 48
        self.convlru_num_blocks = 8
        self.hidden_factor = (10, 20)
        self.emb_hidden_ch = 1
        self.emb_hidden_layers_num = 72
        self.ffn_hidden_ch = 96
        self.ffn_hidden_layers_num = 2
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.out_ch = 24
        self.gen_factor = 1
        self.hidden_activation = 'Tanh'
        self.output_activation = 'Tanh'
        self.data_root = '/nfs/ERA5_data/data_norm'
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 13
        self.eval_data_n_frames = 4
        self.eval_sample_num = 32
        self.ckpt = 'e2_s222_l0.049569.pth'
        self.train_batch_size = 3
        self.eval_batch_size = 3
        self.epochs = 50
        self.log_path = './convlru_base/logs'
        self.ckpt_dir = './convlru_base/ckpt'
        self.ckpt_step = 0.25
        self.do_eval = False
        if not self.do_eval:
            self.eval_sample_num = 1
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-4
        self.use_scheduler = False
        self.init_lr_scheduler = False
        self.loss_latl1_weight = 1.0
        self.loss_ssim_weight = 0.0
        self.ssim_window_size = 11
        self.ssim_sigma = 1.5

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
    files_to_delete = ckpt_files[3:]
    for file_path in files_to_delete:
        os.remove(file_path)

MODEL_ARG_KEYS = [
    'input_size', 'input_ch', 'use_mhsa', 'use_gate',
    'emb_ch', 'convlru_num_blocks', 'hidden_factor',
    'emb_hidden_ch', 'emb_hidden_layers_num',
    'ffn_hidden_ch', 'ffn_hidden_layers_num',
    'dec_hidden_ch', 'dec_hidden_layers_num',
    'out_ch', 'gen_factor',
    'hidden_activation', 'output_activation',
]

def extract_model_args(args_obj):
    d = {}
    for k in MODEL_ARG_KEYS:
        if hasattr(args_obj, k):
            d[k] = getattr(args_obj, k)
    return d

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
    for k in ['model', 'optimizer', 'scheduler']:
        if isinstance(ckpt, dict) and k in ckpt:
            del ckpt[k]
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()
    if not model_args:
        print(f"[Args] Warning: no model_args found in {ckpt_path}, using code defaults.")
        logging.warning(f"[Args] no model_args found in {ckpt_path}, using code defaults.")
        return None
    return model_args

def save_ckpt(model, opt, epoch, step, loss, args, scheduler=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss.item() if hasattr(loss, 'item') else float(loss),
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

def load_ckpt(model, opt, ckpt_path, scheduler=None, map_location='cpu', args=None, restore_model_args=False):
    if os.path.isfile(ckpt_path):
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
        for k in ['model', 'optimizer', 'scheduler']:
            if k in checkpoint:
                del checkpoint[k]
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
        logging.info(f"Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
        return epoch, step
    else:
        print(f"No checkpoint Found")
        logging.warning(f"No checkpoint Found at {ckpt_path}")
        gc.collect()
        torch.cuda.empty_cache()
        return 0, 0

def setup_logging(args):
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    with open(__file__, 'r') as f_script, open(log_filename, 'w') as f_log:
        f_log.write("========== convlru_train_ddp.py ==========\n")
        f_log.write(f_script.read())
        f_log.write("\n========== End of Script ==========\n\n")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

_SSIM_WIN_CACHE = {}
_LAT_WEIGHT_CACHE = {}

def _gaussian_window(window_size=11, sigma=1.5, channels=1, device='cpu', dtype=torch.float32):
    key = (window_size, sigma, channels, device, dtype)
    if key in _SSIM_WIN_CACHE:
        return _SSIM_WIN_CACHE[key]
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = (g @ g.t())
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    _SSIM_WIN_CACHE[key] = window
    return window

def _ssim_map(x, y, window, data_range=1.0, K=(0.01, 0.03)):
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2
    padding = window.shape[-1] // 2
    channels = x.shape[1]
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = numerator / (denominator + 1e-12)
    return ssim

def ssim_value(preds, targets, window_size=11, sigma=1.5):
    assert preds.shape == targets.shape
    B, T, C, H, W = preds.shape
    device = preds.device
    dtype = preds.dtype
    x = (preds + 1.0) * 0.5
    y = (targets + 1.0) * 0.5
    x = x.reshape(B * T, C, H, W)
    y = y.reshape(B * T, C, H, W)
    window = _gaussian_window(window_size=window_size, sigma=sigma, channels=C, device=device, dtype=dtype)
    ssim_map = _ssim_map(x, y, window, data_range=1.0)
    ssim_val = ssim_map.mean()
    return ssim_val

def ssim_loss(preds, targets, window_size=11, sigma=1.5):
    return 1.0 - ssim_value(preds, targets, window_size, sigma)

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
    assert preds.shape == targets.shape
    B, T, C, H, W = preds.shape
    device = preds.device
    dtype = preds.dtype
    w = get_latitude_weights(H, device, dtype).view(1, 1, 1, H, 1)
    diff = (preds - targets).abs()
    loss = (diff * w).mean()
    return loss

def compute_total_loss(pred_slice, targ_slice, args, need_ssim_stats=False):
    latl1_val = latitude_weighted_l1(pred_slice, targ_slice) if args.loss_latl1_weight > 0 else torch.tensor(0.0, device=pred_slice.device)
    if args.loss_ssim_weight > 0:
        ssim_val_loss = ssim_loss(pred_slice, targ_slice, window_size=args.ssim_window_size, sigma=args.ssim_sigma)
        total = args.loss_latl1_weight * latl1_val + args.loss_ssim_weight * ssim_val_loss
        if need_ssim_stats:
            return total, latl1_val, ssim_val_loss
        else:
            return total, latl1_val, None
    else:
        total = args.loss_latl1_weight * latl1_val
        return total, latl1_val, None

def run_ddp(rank, world_size, local_rank, master_addr, master_port, args):
    setup_ddp(rank, world_size, master_addr, master_port, local_rank)
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    if rank == 0:
        setup_logging(args)
        logging.info("========== Training Arguments ==========")
        for k, v in vars(args).items():
            logging.info(f"{k}: {v}")
        logging.info("========================================")
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_model_args = load_model_args_from_ckpt(args.ckpt, map_location=f'cuda:{local_rank}')
        if ckpt_model_args:
            print("[Args] applying model args from ckpt before building model.")
            logging.info("[Args] applying model args from ckpt before building model.")
            apply_model_args(args, ckpt_model_args, verbose=True)
    model = ConvLRU(args)
    model = model.cuda(local_rank)
    if args.use_compile:
        model = torch.compile(model, mode="default")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    train_dataset = ERA5_Dataset(
        input_dir=args.data_root, year_range=args.year_range,
        is_train=True, sample_len=args.train_data_n_frames,
        eval_sample=args.eval_sample_num, max_cache_size=5,
        rank=dist.get_rank(), gpus=dist.get_world_size())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        num_workers=1, pin_memory=True, prefetch_factor=1)
    len_train_dataloader = len(train_dataloader)
    del train_dataset, train_sampler, train_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len_train_dataloader, epochs=len(list(range(start_epoch, args.epochs))))
    if args.ckpt and os.path.isfile(args.ckpt):
        start_epoch, _ = load_ckpt(model, opt, args.ckpt, scheduler, map_location=f'cuda:{local_rank}', args=args, restore_model_args=False)
    if args.init_lr_scheduler and args.use_scheduler:
        print(f"Init lr scheduler.")
        logging.info(f"Init lr scheduler.")
        scheduler = lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len_train_dataloader, epochs=len(list(range(start_epoch, args.epochs))))
    if not args.use_scheduler:
        print(f"Scheduler is disable, opt will init.")
        logging.warning(f"Scheduler is disable, opt will init.")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(start_epoch, args.epochs):
        train_dataset = ERA5_Dataset(
            input_dir=args.data_root, year_range=args.year_range,
            is_train=True, sample_len=args.train_data_n_frames,
            eval_sample=args.eval_sample_num, max_cache_size=5,
            rank=dist.get_rank(), gpus=dist.get_world_size())
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
            num_workers=1, pin_memory=True, prefetch_factor=2)
        train_sampler.set_epoch(ep)
        train_dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{args.epochs} - Start") if rank == 0 else train_dataloader
        for train_step, data in enumerate(train_dataloader_iter, start=1):
            model.train()
            opt.zero_grad()
            data = data.cuda(local_rank).to(torch.float32)[:, :, :, 1:, :]
            preds = model(data[:, :-1], 'p')
            pred_slice = preds[:, 1:]
            targ_slice = data[:, 2:]
            loss, latl1_val, ssim_val_loss = compute_total_loss(pred_slice, targ_slice, args, need_ssim_stats=True)
            loss.backward()
            opt.step()
            if args.use_scheduler:
                scheduler.step()
            loss_tensor = torch.tensor(loss.detach().item(), device=f'cuda:{local_rank}')
            latl1_tensor = torch.tensor(latl1_val.detach().item(), device=f'cuda:{local_rank}')
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(latl1_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size
            avg_latl1 = latl1_tensor.item() / world_size
            if ssim_val_loss is not None:
                ssiml_tensor = torch.tensor(ssim_val_loss.detach().item(), device=f'cuda:{local_rank}')
                dist.all_reduce(ssiml_tensor, op=dist.ReduceOp.SUM)
                avg_ssim = 1.0 - (ssiml_tensor.item() / world_size)
            else:
                avg_ssim = None
            if ssim_val_loss is not None:
                del ssim_val_loss
            if rank == 0:
                if args.use_scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                if not args.use_scheduler:
                    current_lr = args.lr
                if avg_ssim is not None:
                    message = f"Epoch {ep+1}/{args.epochs} - Step {train_step} - Total: {avg_loss:.6f} - LatL1: {avg_latl1:.6f} - SSIM: {avg_ssim:.6f} (w_latl1={args.loss_latl1_weight:.3f}, w_ssim={args.loss_ssim_weight:.3f}) - LR: {current_lr:.6e}"
                else:
                    message = f"Epoch {ep+1}/{args.epochs} - Step {train_step} - Total: {avg_loss:.6f} - LatL1: {avg_latl1:.6f} (w_latl1={args.loss_latl1_weight:.3f}, w_ssim={args.loss_ssim_weight:.3f}) - LR: {current_lr:.6e}"
                train_dataloader_iter.set_description(message)
                logging.info(message)
            if rank == 0 and (train_step % int(len(train_dataloader)*args.ckpt_step) == 0 or train_step == len(train_dataloader)):
                save_ckpt(model, opt, ep, train_step, avg_loss, args, scheduler)
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
                    data = data.cuda(local_rank).to(torch.float32)[:, :, :, 1:, :]
                    out_gen_num = data[:, args.eval_data_n_frames//2:].shape[1] // args.gen_factor
                    preds = model(data[:, :args.eval_data_n_frames//2], 'i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
                    if args.loss_ssim_weight > 0:
                        loss_total_eval, loss_latl1_eval, loss_ssim_eval = compute_total_loss(preds, data[:, args.eval_data_n_frames//2:], args, need_ssim_stats=True)
                        tot_tensor = torch.tensor(loss_total_eval.item(), device=f'cuda:{local_rank}')
                        latl1_tensor = torch.tensor(loss_latl1_eval.item(), device=f'cuda:{local_rank}')
                        ssiml_tensor = torch.tensor(loss_ssim_eval.detach().item(), device=f'cuda:{local_rank}')
                        dist.all_reduce(tot_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(latl1_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(ssiml_tensor, op=dist.ReduceOp.SUM)
                        avg_total = tot_tensor.item() / world_size
                        avg_latl1 = latl1_tensor.item() / world_size
                        avg_ssim = 1.0 - (ssiml_tensor.item() / world_size)
                    else:
                        loss_total_eval, loss_latl1_eval, _ = compute_total_loss(preds, data[:, args.eval_data_n_frames//2:], args, need_ssim_stats=False)
                        tot_tensor = torch.tensor(loss_total_eval.item(), device=f'cuda:{local_rank}')
                        latl1_tensor = torch.tensor(loss_latl1_eval.item(), device=f'cuda:{local_rank}')
                        dist.all_reduce(tot_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(latl1_tensor, op=dist.ReduceOp.SUM)
                        avg_total = tot_tensor.item() / world_size
                        avg_latl1 = latl1_tensor.item() / world_size
                        avg_ssim = None
                    if rank == 0:
                        if avg_ssim is not None:
                            message = f"Eval step {eval_step} - Total: {avg_total:.6f} - LatL1: {avg_latl1:.6f} - SSIM: {avg_ssim:.6f} (w_latl1={args.loss_latl1_weight:.3f}, w_ssim={args.loss_ssim_weight:.3f})"
                        else:
                            message = f"Eval step {eval_step} - Total: {avg_total:.6f} - LatL1: {avg_latl1:.6f} (w_latl1={args.loss_latl1_weight:.3f}, w_ssim={args.loss_ssim_weight:.3f})"
                        eval_dataloader_iter.set_description(message)
                        logging.info(message)
                del eval_dataset, eval_sampler, eval_dataloader, eval_dataloader_iter
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')
    run_ddp(rank, world_size, local_rank, master_addr, master_port, args)
