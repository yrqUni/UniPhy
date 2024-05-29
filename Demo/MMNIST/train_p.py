import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Model'))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import heapq

from ModelConvLRU import ConvLRU
from DATA.MMNIST import MovingMNIST

class Args:
    def __init__(self):
        # input info
        self.input_size = (64, 64)
        self.input_ch = 1
        # convlru info
        self.emb_ch = 256
        self.convlru_dropout = 0.0
        self.convlru_num_blocks = 24
        #
        self.hidden_factor = (2, 2)
        # emb info
        self.emb_hidden_ch = 64
        self.emb_dropout = 0.0
        self.emb_hidden_layers_num = 4
        # ffn info
        self.ffn_hidden_ch = 32
        self.ffn_dropout = 0.0
        self.ffn_hidden_layers_num = 2
        # dec info
        self.dec_hidden_ch = 64
        self.dec_dropout = 0.0
        self.dec_hidden_layers_num = 4
        # training data info
        self.root = './DATA/MMNIST/'
        self.is_train = True
        self.n_frames_input = 65
        self.n_frames_output = 1
        self.num_objects = [2]
        self.num_samples = int(1e4)
        # training info
        self.batch_size = 2
        self.lr = 1e-3
        self.EPs = 500
        self.vis_step = 100
        self.vis_num = 10
        self.val_step = 500
        self.out_path = './exp0/train/'
        self.log_file = os.path.join(self.out_path, 'log')
        self.ckpt_path = os.path.join(self.out_path, 'ckpt/')
        self.vis_path = os.path.join(self.out_path, 'vis/')
        self.pretrain_path = 'None'
        self.ckpt_num = 5  
        # evaluation data info
        self.eval_root = './DATA/MMNIST/'
        self.eval_is_train = True
        self.eval_n_frames_input = 10
        self.eval_n_frames_output = 10
        self.eval_num_objects = [2]
        self.eval_num_samples = int(1e2)
        # evaluation info
        self.eval_batch_size = 2
        self.eval_vis_num = 10
        self.eval_out_path = './exp0/eval/'
        self.eval_ckpt_path = os.path.join(self.eval_out_path, 'ckpt/')
        self.eval_pretrain_path = 'None'
        # random seed
        self.seed = 1017
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

args = Args()

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
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(str(args))
    print(str(args))

def visualize(GTs, PREDs, epoch, step, vis_num, path):
    L = GTs.size(1)
    GTs, PREDs = GTs.squeeze(2), PREDs.squeeze(2)
    indices = random.sample(range(L), min(vis_num, L))
    _, axes = plt.subplots(2, vis_num, figsize=(20, 5))
    for i, idx in enumerate(indices):
        axes[0, i].imshow(GTs[0, idx].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"GT {idx}")
        axes[0, i].axis('off')
        axes[1, i].imshow(torch.sigmoid(PREDs[0, idx]).cpu().detach().numpy(), cmap='gray')
        axes[1, i].set_title(f"Pred {idx}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'visualization_epoch{epoch}_step{step}.png'))
    plt.close()

def evaluate_model(model, dataloader, criterion, args, epoch, step):
    model.eval()
    ssim_vals, mse_vals, bce_vals = [], [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, mode='i_sigmoid', out_frames_num=args.eval_n_frames_output)
            bce_loss = criterion(outputs, targets)
            bce_vals.append(bce_loss.item())
            for b in range(outputs.size(0)):
                for i in range(outputs.size(1)):
                    ssim_val = ssim(targets[b, i].cpu().numpy(), outputs[b, i].cpu().numpy(), data_range=1.0, win_size=5, channel_axis=0)
                    ssim_vals.append(ssim_val)
                    mse_vals.append(mean_squared_error(targets[b, i].cpu().numpy().ravel(), outputs[b, i].cpu().numpy().ravel()))
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

    dataset = MovingMNIST(root=args.root, is_train=args.is_train, n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output, num_objects=args.num_objects, num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    eval_dataset = MovingMNIST(root=args.eval_root, is_train=args.eval_is_train, n_frames_input=args.eval_n_frames_input, n_frames_output=args.eval_n_frames_output, num_objects=args.eval_num_objects, num_samples=args.eval_num_samples)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    model = ConvLRU(args).cuda()

    if os.path.exists(args.pretrain_path):
        model.load_state_dict(torch.load(args.pretrain_path))
        logging.info(f'Loaded pretrained model from {args.pretrain_path}')
    else:
        logging.info('No pretrained model found, starting from scratch.')

    criterion = nn.BCEWithLogitsLoss().cuda()
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataloader))

    best_val_ssim = float('-inf')
    for ep in range(args.EPs):
        model.train()
        running_loss = 0.0
        for step, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.cuda()
            targets = targets.cuda()
            opt.zero_grad()
            pred_outputs = model(inputs[:, :-1], mode='p_logits')
            loss = criterion(pred_outputs[:, 1:], inputs[:, 2:])
            loss.backward()
            opt.step()
            scheduler.step()
            running_loss += loss.item()

            if (step + 1) % args.vis_step == 0:
                avg_loss = running_loss / args.vis_step
                current_lr = scheduler.get_last_lr()[0]
                logging.info(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, LR: {current_lr}')
                checkpoint_path = save_checkpoint(args.ckpt_path, model, ep, step + 1)
                manage_checkpoints(args.ckpt_path, args.ckpt_num)
                tqdm.write(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, LR: {current_lr}')
                running_loss = 0.0
                visualize(inputs[:, 1:], pred_outputs, ep, step + 1, args.vis_num, args.vis_path)

            if (step + 1) % args.val_step == 0:
                mean_ssim, mean_mse, mean_bce = evaluate_model(model, eval_dataloader, criterion, args, ep, step + 1)
                if mean_ssim > best_val_ssim:
                    best_val_ssim = mean_ssim
                    save_checkpoint(args.eval_ckpt_path, model, ep, step + 1, best=True)
                    logging.info(f'New best model at Epoch {ep}, Step {step+1} with SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
                    tqdm.write(f'New best model at Epoch {ep}, Step {step+1} with SSIM={mean_ssim}, MSE={mean_mse}, BCE={mean_bce}')
                manage_checkpoints(args.eval_ckpt_path, args.ckpt_num)

    logging.shutdown()

if __name__ == "__main__":
    train_model(args)
