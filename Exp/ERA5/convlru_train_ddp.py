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
from ModelConvLRU_ERA5 import ConvLRU
from ERA5 import ERA5_Dataset
from tqdm import tqdm

class Args:
    def __init__(self):
        self.input_size = (720, 1440)
        self.input_ch = 89
        self.use_mhsa = True
        self.use_gate = False
        self.emb_ch = 128
        self.convlru_num_blocks = 12
        self.hidden_factor = (10, 20)
        self.emb_hidden_ch = 256
        self.emb_hidden_layers_num = 1
        self.ffn_hidden_ch = 256
        self.ffn_hidden_layers_num = 2
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.out_ch = 89
        self.gen_factor = 1
        self.hidden_activation = 'Tanh'
        self.output_activation = 'Tanh'
        self.data_root = '/nfs/ERA5_data/data_norm'
        self.year_range = [2000, 2021]
        self.train_data_n_frames = 15
        self.eval_data_n_frames = 4
        self.eval_sample_num = 32
        self.ckpt = ''
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.epochs = 100
        self.log_path = './convlru_base/logs'
        self.ckpt_dir = './convlru_base/ckpt'
        self.ckpt_step = 0.25
        self.do_eval = False
        if not self.do_eval:
            self.eval_sample_num = 1
        self.use_tf32 = False
        self.use_compile = False
        self.lr = 1e-3
        self.init_lr_scheduler = True
        self.use_scheduler = True

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

def save_ckpt(model, opt, epoch, step, loss, args, scheduler=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    state = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss.item() if hasattr(loss, 'item') else float(loss),
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

def load_ckpt(model, opt, ckpt_path, scheduler=None, map_location='cpu'):
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=map_location)
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
    model = ConvLRU(args)
    model = model.cuda(local_rank)
    if args.use_compile:
        model = torch.compile(model, mode="default")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    loss_fn = torch.nn.MSELoss().cuda(local_rank)
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
        start_epoch, _ = load_ckpt(model, opt, args.ckpt, scheduler, map_location=f'cuda:{local_rank}')
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
            num_workers=1, pin_memory=True, prefetch_factor=1)
        train_sampler.set_epoch(ep)
        train_dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{args.epochs} - Start") if rank == 0 else train_dataloader
        for train_step, data in enumerate(train_dataloader_iter, start=1):
            model.train()
            opt.zero_grad()
            data = data.cuda(local_rank).to(torch.float32)
            inputs, outputs = data[:, :-1], data[:, 1:]
            del data
            gc.collect()
            torch.cuda.empty_cache()
            preds = model(inputs, 'p')
            loss = loss_fn(preds[:, 1:], outputs[:, 1:])
            loss.backward()
            opt.step()
            if args.use_scheduler:
                scheduler.step()
            loss_value = loss.item()
            loss_tensor = torch.tensor(loss_value).cuda(local_rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size
            del inputs, outputs, preds, loss, loss_tensor
            gc.collect()
            torch.cuda.empty_cache()
            if train_step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if rank == 0:
                if args.use_scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                if not args.use_scheduler:
                    current_lr = args.lr
                message = f"Epoch {ep+1}/{args.epochs} - Step {train_step} - Loss: {avg_loss:.6f} - LR: {current_lr:.6e}"
                train_dataloader_iter.set_description(message)
                logging.info(message)
            if rank == 0 and (train_step % int(len(train_dataloader)*args.ckpt_step) == 0 or train_step == len(train_dataloader)):
                save_ckpt(model, opt, ep, train_step, avg_loss, args, scheduler)
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
                loss_value = 0
                for eval_step, data in enumerate(eval_dataloader_iter, start=1):
                    data = data.cuda(local_rank).to(torch.float32)
                    inputs, outputs = data[:, :args.eval_data_n_frames//2], data[:, args.eval_data_n_frames//2:]
                    out_gen_num = outputs.shape[1] // args.gen_factor
                    preds = model(inputs, 'i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
                    loss = loss_fn(preds, outputs)
                    loss_value += loss.item()
                    _loss_value = loss_value / eval_step
                    loss_tensor = torch.tensor(_loss_value).cuda(local_rank)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss = loss_tensor.item() / world_size
                    del inputs, outputs, preds, loss, loss_tensor, data
                    gc.collect()
                    torch.cuda.empty_cache()
                    if rank == 0:
                        message = f"Eval step {eval_step} - Eval loss: {avg_loss:.6f}"
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

###############
# END OF FILE #
###############

