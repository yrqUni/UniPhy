import sys
import os
sys.path.append('../../../Model')
sys.path.append('../DATA')

import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM

from ModelConvLRU import ConvLRU
from DATA.MMNIST import MovingMNIST

class Args:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.input_size = tuple(config.get("input_size", [64, 64]))
        self.input_ch = config.get("input_ch", 1)
        self.emb_ch = config.get("emb_ch", 256)
        self.convlru_num_blocks = config.get("convlru_num_blocks", 24)
        self.hidden_factor = tuple(config.get("hidden_factor", [2, 2]))
        self.emb_hidden_ch = config.get("emb_hidden_ch", 128)
        self.emb_hidden_layers_num = config.get("emb_hidden_layers_num", 22)
        self.ffn_hidden_ch = config.get("ffn_hidden_ch", 64)
        self.ffn_hidden_layers_num = config.get("ffn_hidden_layers_num", 12)
        self.dec_hidden_ch = config.get("dec_hidden_ch", 128)
        self.dec_hidden_layers_num = config.get("dec_hidden_layers_num", 22)
        self.gen_factor = config.get("gen_factor", 16)
        self.output_activation = config.get("hidden_activation", 'ReLU')
        self.output_activation = config.get("output_activation", 'Sigmoid')
        self.root = config.get("root", './DATA/MMNIST/')
        self.is_train = config.get("is_train", True)
        self.n_frames_input = config.get("n_frames_input", 21)
        self.n_frames_output = config.get("n_frames_output", 1)
        self.num_objects = config.get("num_objects", [2])
        self.num_samples = config.get("num_samples", int(1e4))
        self.out_path_root = config.get("out_path_root", './exp1/')
        self.batch_size = config.get("batch_size", 5)
        self.lr = config.get("lr", 1e-3)
        self.EPs = config.get("EPs", 500)
        self.vis_step = config.get("vis_step", 100)
        self.vis_num = config.get("vis_num", 10)
        self.val_step = config.get("val_step", 1000)
        self.out_path = config.get("out_path", os.path.join(self.out_path_root, 'train'))
        self.log_file = config.get("log_file", os.path.join(self.out_path, 'log'))
        self.ckpt_path = config.get("ckpt_path", os.path.join(self.out_path, 'ckpt/'))
        self.vis_path = config.get("vis_path", os.path.join(self.out_path, 'vis/'))
        self.pretrain_path = config.get("pretrain_path", 'None')
        self.ckpt_num = config.get("ckpt_num", 5)
        self.eval_root = config.get("eval_root", './DATA/MMNIST/')
        self.eval_is_train = config.get("eval_is_train", False)
        self.eval_n_frames_input = config.get("eval_n_frames_input", 10)
        self.eval_n_frames_output = config.get("eval_n_frames_output", 10)
        self.eval_num_objects = config.get("eval_num_objects", [2])
        self.eval_num_samples = config.get("eval_num_samples", int(1e2))
        self.eval_batch_size = config.get("eval_batch_size", 5)
        self.eval_vis_num = config.get("eval_vis_num", 10)
        self.eval_out_path = config.get("eval_out_path", os.path.join(self.out_path_root, 'eval'))
        self.eval_ckpt_path = config.get("eval_ckpt_path", os.path.join(self.eval_out_path, 'ckpt/'))
        self.seed = config.get("seed", 1017)
    def __str__(self):
        attrs = vars(self)
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)
    if not os.path.exists(args.eval_out_path):
        os.makedirs(args.eval_out_path)
    if not os.path.exists(args.eval_ckpt_path):
        os.makedirs(args.eval_ckpt_path)

def initialize_logging(args):
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(str(args))
    print(str(args))

def visualize(GTs, PREDs, epoch, step, vis_num, path):
    B, L, C, H, W = GTs.size()
    idx_B = np.random.randint(B)
    GTs, PREDs = GTs[idx_B], PREDs[idx_B]
    indices = np.linspace(0, L-1, vis_num, dtype=int)

    fig, axes = plt.subplots(C * 2, vis_num, figsize=(20, 10 * C))

    for c in range(C):
        for i, idx in enumerate(indices):
            axes[c * 2, i].imshow(GTs[idx, c].cpu().numpy(), cmap='gray')
            axes[c * 2, i].set_title(f"GT {idx} Channel {c}")
            axes[c * 2, i].axis('off')

            axes[c * 2 + 1, i].imshow(torch.sigmoid(PREDs[idx, c]).cpu().detach().numpy(), cmap='gray')
            axes[c * 2 + 1, i].set_title(f"Pred {idx} Channel {c}")
            axes[c * 2 + 1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'visualization_epoch{epoch}_step{step}.png'))
    plt.close()

def evaluate_model(model, dataloader, args, epoch, step):
    criterion = nn.BCELoss().cuda()
    ssim_criterion = SSIM(data_range=1.0, size_average=True).cuda()
    model.eval()
    ssim_vals, mse_vals, bce_vals = [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.cuda(), targets.cuda()
            out_gen_num = args.eval_n_frames_output // args.gen_factor
            outputs = model(inputs, mode='i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
            B, L, C, H, W = outputs.size()
            bce_loss = criterion(outputs, targets)
            ssim_val = ssim_criterion(outputs.reshape(B*L, C, H, W).expand(B*L, 3, H, W), targets.reshape(B*L, C, H, W).expand(B*L, 3, H, W)).item()
            bce_vals.append(bce_loss.item())
            ssim_vals.append(ssim_val)
            mse_fn = nn.MSELoss().cuda()
            mse_vals.append(mse_fn(outputs, targets).item())
    mean_ssim = sum(ssim_vals) / len(ssim_vals)
    mean_mse = sum(mse_vals) / len(mse_vals)
    mean_bce = sum(bce_vals) / len(bce_vals)
    logging.info(f'Evaluation at Epoch {epoch}, Step {step}: SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
    print(f'Evaluation at Epoch {epoch}, Step {step}: SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
    return mean_ssim, mean_mse, mean_bce

def save_checkpoint(ckpt_path, model, epoch, step, best=False):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    filename = f'{"best_" if best else ""}checkpoint_epoch{epoch}_step{step}.pth'
    filepath = os.path.join(ckpt_path, filename)
    torch.save(model.state_dict(), filepath)
    return filepath

def manage_checkpoints(ckpt_path, max_ckpt_num):
    checkpoints = [f for f in os.listdir(ckpt_path) if f.endswith('.pth')]
    if len(checkpoints) > max_ckpt_num:
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
        for ckpt in checkpoints[:-max_ckpt_num]:
            os.remove(os.path.join(ckpt_path, ckpt))

def train_model(args):
    set_random_seed(args.seed)
    create_directories(args)
    initialize_logging(args)

    eval_dataset = MovingMNIST(root=args.eval_root, is_train=args.eval_is_train, n_frames_input=args.eval_n_frames_input, n_frames_output=args.eval_n_frames_output, num_objects=args.eval_num_objects, num_samples=args.eval_num_samples)
    if len(eval_dataset) > args.eval_num_samples:
        indices = np.random.choice(len(eval_dataset), args.eval_num_samples, replace=False)
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)

    model = ConvLRU(args).cuda()

    if os.path.exists(args.pretrain_path):
        model.load_state_dict(torch.load(args.pretrain_path))
        logging.info(f'Loaded pretrained model from {args.pretrain_path}')
    else:
        logging.info('No pretrained model found, starting from scratch.')

    criterion = nn.MSELoss().cuda()
    ssim_criterion = SSIM(data_range=1.0, size_average=True).cuda()
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_ssim = float('-inf')
    for ep in range(args.EPs):
        dataset = MovingMNIST(root=args.root, is_train=args.is_train, n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output, num_objects=args.num_objects, num_samples=args.num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)
        max_lr = args.lr
        scheduler = OneCycleLR(opt, max_lr=max_lr, steps_per_epoch=len(dataloader), epochs=args.EPs)

        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        for step, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.cuda()
            # targets = targets.cuda()
            opt.zero_grad()
            pred_outputs = model(inputs[:, :-args.gen_factor], mode='p')
            sigmoid_pred_outputs = torch.sigmoid(pred_outputs)
            mse_loss = criterion(sigmoid_pred_outputs, inputs[:, args.gen_factor:])
            B, L, C, H, W = sigmoid_pred_outputs.size()
            ssim_values = ssim_criterion(sigmoid_pred_outputs.reshape(B*L, C, H, W).expand(B*L, 3, H, W), inputs[:, args.gen_factor:].reshape(B*L, C, H, W).expand(B*L, 3, H, W))
            total_loss = mse_loss + (1 - ssim_values)
            total_loss.backward()
            opt.step()
            scheduler.step()
            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_ssim += ssim_values.item()

            if (step + 1) % args.vis_step == 0:
                avg_loss = running_loss / args.vis_step
                avg_mse = running_mse / args.vis_step
                avg_ssim = running_ssim / args.vis_step
                current_lr = scheduler.get_last_lr()[0]
                logging.info(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, Average MSE: {avg_mse}, Average SSIM: {avg_ssim}, LR: {current_lr}')
                checkpoint_path = save_checkpoint(args.ckpt_path, model, ep, step + 1)
                manage_checkpoints(args.ckpt_path, args.ckpt_num)
                tqdm.write(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, Average MSE: {avg_mse}, Average SSIM: {avg_ssim}, LR: {current_lr}')
                running_loss = 0.0
                running_mse = 0.0
                running_ssim = 0.0
                visualize(inputs[:, args.gen_factor:], pred_outputs, ep, step + 1, args.vis_num, args.vis_path)

            if (step + 1) % args.val_step == 0:
                mean_ssim, mean_mse, mean_bce = evaluate_model(model, eval_dataloader, args, ep, step + 1)
                if mean_ssim > best_val_ssim:
                    best_val_ssim = mean_ssim
                    save_checkpoint(args.eval_ckpt_path, model, ep, step + 1, best=True)
                    logging.info(f'New best model at Epoch {ep}, Step {step+1} with SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
                    tqdm.write(f'New best model at Epoch {ep}, Step {step+1} with SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
                manage_checkpoints(args.eval_ckpt_path, args.ckpt_num)

    logging.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load config file")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config_file = args.cfg
    args = Args(config_file)
    train_model(args)

