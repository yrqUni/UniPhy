import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import wandb

from ModelConvLRU import ConvLRU
from ERA5 import ERA5_Dataset

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return rank, local_rank, world_size
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456", world_size=1, rank=0)
        return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser("ConvLRU Training", add_help=False)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--accum_iter", default=1, type=int)
    
    parser.add_argument("--input_ch", default=20, type=int)
    parser.add_argument("--out_ch", default=20, type=int)
    parser.add_argument("--input_size", default=(64, 128), type=int, nargs="+")
    parser.add_argument("--emb_ch", default=128, type=int)
    parser.add_argument("--emb_hidden_ch", default=256, type=int)
    parser.add_argument("--emb_hidden_layers_num", default=2, type=int)
    parser.add_argument("--static_ch", default=3, type=int)
    parser.add_argument("--hidden_factor", default=(2, 2), type=int, nargs="+")
    
    parser.add_argument("--convlru_num_blocks", default=4, type=int)
    parser.add_argument("--ffn_hidden_ch", default=256, type=int)
    parser.add_argument("--ffn_hidden_layers_num", default=2, type=int)
    parser.add_argument("--use_cbam", action="store_true")
    parser.add_argument("--lru_rank", default=16, type=int)
    parser.add_argument("--use_selective", action="store_true")
    parser.add_argument("--use_gate", action="store_true")
    
    parser.add_argument("--num_expert", default=4, type=int)
    parser.add_argument("--activate_expert", default=2, type=int)
    
    parser.add_argument("--use_freq_prior", action="store_true")
    parser.add_argument("--use_sh_prior", action="store_true")
    parser.add_argument("--sh_Lmax", default=6, type=int)
    parser.add_argument("--sh_rank", default=8, type=int)
    parser.add_argument("--sh_gain_init", default=0.0, type=float)
    
    parser.add_argument("--use_spectral_mixing", action="store_true")
    parser.add_argument("--use_anisotropic_diffusion", action="store_true")
    parser.add_argument("--use_advection", action="store_true")
    parser.add_argument("--learnable_init_state", action="store_true")
    parser.add_argument("--use_wavelet_ssm", action="store_true")
    parser.add_argument("--use_cross_var_attn", action="store_true")
    
    parser.add_argument("--ConvType", default="conv", type=str)
    parser.add_argument("--Arch", default="unet", type=str)
    
    parser.add_argument("--head_mode", default="gaussian", type=str)
    parser.add_argument("--dec_hidden_ch", default=128, type=int)
    parser.add_argument("--dec_hidden_layers_num", default=2, type=int)
    parser.add_argument("--dec_strategy", default="pxsf", type=str)
    
    parser.add_argument("--unet", action="store_true")
    parser.add_argument("--down_mode", default="avg", type=str)
    
    parser.add_argument("--train_data_path", default="./data/train.joblib", type=str)
    parser.add_argument("--valid_data_path", default="./data/valid.joblib", type=str)
    parser.add_argument("--static_data_path", default="./data/static.pt", type=str)
    parser.add_argument("--train_data_n_frames", default=10, type=int)
    parser.add_argument("--output_dir", default="./checkpoints", type=str)
    parser.add_argument("--wandb_project", default="ConvLRU_ERA5", type=str)
    
    return parser

def gaussian_nll_loss_weighted(pred, target, weight=None):
    mu, sigma = torch.chunk(pred, 2, dim=2)
    loss = 0.5 * torch.log(sigma**2) + 0.5 * (target - mu)**2 / (sigma**2)
    if weight is not None:
        loss = loss * weight
    return loss.mean()

def latitude_weighted_l1(pred, target, lat_weight=None):
    diff = torch.abs(pred - target)
    if lat_weight is not None:
        diff = diff * lat_weight
    return diff.mean()

def gradient_difference_loss(pred, target):
    dy_true = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
    dx_true = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dx_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    loss = torch.mean(torch.abs(dy_true - dy_pred)) + torch.mean(torch.abs(dx_true - dx_pred))
    return loss

def spectral_loss(pred, target):
    pred_fft = torch.fft.rfftn(pred, dim=(-2, -1))
    target_fft = torch.fft.rfftn(target, dim=(-2, -1))
    loss = torch.mean(torch.abs(pred_fft - target_fft))
    return loss

def get_moe_aux_loss(model):
    aux_loss = 0.0
    for name, module in model.named_modules():
        if hasattr(module, "aux_loss"):
            aux_loss += module.aux_loss
    return aux_loss

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_ddp()
    
    if rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        wandb.init(project=args.wandb_project, config=args)
    
    device = torch.device(f"cuda:{local_rank}")
    
    static_feats = None
    if os.path.exists(args.static_data_path):
        static_feats = torch.load(args.static_data_path, map_location=device)
        args.static_ch = static_feats.shape[1]
    
    dataset_train = ERA5_Dataset(args.train_data_path, n_frames=args.train_data_n_frames)
    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.num_workers, pin_memory=True)
    
    dataset_valid = ERA5_Dataset(args.valid_data_path, n_frames=args.train_data_n_frames)
    sampler_valid = DistributedSampler(dataset_valid, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, sampler=sampler_valid, num_workers=args.num_workers, pin_memory=True)
    
    model = ConvLRU(args).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    H, W = args.input_size
    lat_weight = torch.cos(torch.linspace(-math.pi/2, math.pi/2, H)).view(1, 1, 1, H, 1).to(device)
    
    for epoch in range(args.epochs):
        model.train()
        sampler_train.set_epoch(epoch)
        optimizer.zero_grad()
        
        train_loss_acc = 0.0
        
        for step, batch in enumerate(dataloader_train):
            x = batch.to(device, non_blocking=True)
            
            # x: [B, T, C, H, W]
            input_seq = x
            target_seq = x 
            
            if args.head_mode in ["gaussian", "token"]:
                if input_seq.shape[1] > 1:
                    input_x = input_seq[:, :-1]
                    target_y = target_seq[:, 1:]
                else:
                    input_x = input_seq
                    target_y = target_seq
            else:
                input_x = input_seq
                target_y = target_seq 

            B_curr, L_curr = input_x.shape[:2]
            timestep = None
            
            with autocast():
                main_loss = 0.0
                
                if args.head_mode == "flow":
                    timestep = torch.rand(B_curr, device=device)
                    noise = torch.randn_like(target_y)
                    t_view = timestep.view(B_curr, 1, 1, 1, 1)
                    x_t = (1 - (1 - 1e-5) * t_view) * noise + t_view * target_y
                    flow_target = target_y - (1 - 1e-5) * noise
                    
                    pred_flow = model(input_x, mode="p", static_feats=static_feats, timestep=timestep)
                    
                    if pred_flow.size(2) == model.module.revin.num_features:
                         pred_flow = model.module.revin(pred_flow, "denorm")
                         
                    main_loss = F.mse_loss(pred_flow, flow_target)
                
                else:
                    preds = model(input_x, mode="p", static_feats=static_feats)
                    
                    if args.head_mode == "gaussian":
                         main_loss = gaussian_nll_loss_weighted(preds, target_y, weight=lat_weight)
                         pred_det = preds[:, :, :args.out_ch]
                    else:
                         main_loss = latitude_weighted_l1(preds, target_y, lat_weight)
                         pred_det = preds
                
                kl_loss = model.module.get_total_kl_loss()
                moe_loss = get_moe_aux_loss(model)
                
                loss_gdl = 0.0
                loss_spec = 0.0
                if args.head_mode != "flow":
                    loss_gdl = gradient_difference_loss(pred_det, target_y)
                    loss_spec = spectral_loss(pred_det, target_y)
                
                total_loss = main_loss + 0.1 * moe_loss + 1e-6 * kl_loss
                if args.head_mode != "flow":
                    total_loss += 0.5 * loss_gdl + 0.1 * loss_spec
            
            scaler.scale(total_loss).backward()
            
            if (step + 1) % args.accum_iter == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            train_loss_acc += total_loss.item()
            
            if rank == 0 and step % 100 == 0:
                wandb.log({
                    "train_loss": total_loss.item(),
                    "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
                    "moe_loss": moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss,
                    "epoch": epoch
                })
                print(f"Epoch {epoch} Step {step} Loss: {total_loss.item():.4f}")

        scheduler.step()
        
        if rank == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.module.state_dict(), save_path)

        model.eval()
        valid_loss_acc = 0.0
        with torch.no_grad():
            for batch in dataloader_valid:
                x = batch.to(device, non_blocking=True)
                input_x = x[:, :-1]
                target_y = x[:, 1:]
                
                if args.head_mode == "flow":
                    timestep = torch.ones(x.shape[0], device=device) * 0.5 
                    preds = model(input_x, mode="p", static_feats=static_feats, timestep=timestep)
                    if preds.size(2) == model.module.revin.num_features:
                         preds = model.module.revin(preds, "denorm")
                    loss = F.mse_loss(preds, target_y) 
                else:
                    preds = model(input_x, mode="p", static_feats=static_feats)
                    if args.head_mode == "gaussian":
                         loss = gaussian_nll_loss_weighted(preds, target_y)
                    else:
                         loss = F.l1_loss(preds, target_y)
                
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                valid_loss_acc += loss.item()
        
        if rank == 0:
            avg_valid_loss = valid_loss_acc / len(dataloader_valid)
            wandb.log({"valid_loss": avg_valid_loss, "epoch": epoch})
            print(f"Epoch {epoch} Valid Loss: {avg_valid_loss:.4f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()

