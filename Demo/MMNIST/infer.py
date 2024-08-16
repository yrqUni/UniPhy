import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Model'))

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models.video import r3d_18

from ModelConvLRU import ConvLRU 
from DATA.MMNIST import MovingMNIST 

class Args:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        # Model
        self.input_size = tuple(config.get("input_size", [64, 64]))
        self.input_ch = config.get("input_ch", 1)
        self.emb_ch = config.get("emb_ch", 256)
        self.convlru_dropout = config.get("convlru_dropout", 0.0)
        self.convlru_num_blocks = config.get("convlru_num_blocks", 24)
        self.hidden_factor = tuple(config.get("hidden_factor", [2, 2]))
        self.emb_hidden_ch = config.get("emb_hidden_ch", 128)
        self.emb_dropout = config.get("emb_dropout", 0.0)
        self.emb_hidden_layers_num = config.get("emb_hidden_layers_num", 22)
        self.ffn_hidden_ch = config.get("ffn_hidden_ch", 64)
        self.ffn_dropout = config.get("ffn_dropout", 0.0)
        self.ffn_hidden_layers_num = config.get("ffn_hidden_layers_num", 12)
        self.dec_hidden_ch = config.get("dec_hidden_ch", 128)
        self.dec_dropout = config.get("dec_dropout", 0.0)
        self.dec_hidden_layers_num = config.get("dec_hidden_layers_num", 22)
        # Data
        self.out_path_root = config.get("out_path_root", './exp1/')
        self.pretrain_path = config.get("pretrain_path", 'None')
        self.eval_root = config.get("eval_root", './DATA/MMNIST/')
        self.eval_is_train = config.get("eval_is_train", False)
        self.eval_n_frames_input = config.get("eval_n_frames_input", 10)
        self.eval_n_frames_output = config.get("eval_n_frames_output", 10)
        self.eval_num_objects = config.get("eval_num_objects", [2])
        self.eval_num_samples = config.get("eval_num_samples", int(1e2))
        self.eval_batch_size = config.get("eval_batch_size", 5)
        self.log_file = config.get("log_file", os.path.join(self.out_path_root, 'log'))
        self.vis_path = config.get("vis_path", os.path.join(self.out_path_root, 'vis/'))
        self.eval_vis_num = config.get("eval_vis_num", 10)
        self.seed = config.get("seed", 1017)
    def __str__(self):
        attrs = vars(self)
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())

parser = argparse.ArgumentParser(description="Load config file")
parser.add_argument('--cfg', type=str, required=True, help="Path to the config file")
args = parser.parse_args()
config_file = args.cfg
args = Args(config_file)

if not os.path.exists(args.vis_path):
    os.makedirs(args.vis_path)

logging.basicConfig(filename=args.log_file, level=logging.INFO)
logging.info(str(args))
print(str(args))

dataset = MovingMNIST(root=args.eval_root, is_train=args.eval_is_train, n_frames_input=args.eval_n_frames_input, n_frames_output=args.eval_n_frames_output, num_objects=args.eval_num_objects, num_samples=args.eval_num_samples)
dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

model = ConvLRU(args).cuda()

if os.path.exists(args.pretrain_path):
    model.load_state_dict(torch.load(args.pretrain_path))
    logging.info(f'Loaded pretrained model from {args.pretrain_path}')
else:
    assert False, 'No pretrained model found'

loss_fn = nn.MSELoss().cuda()

lpips_fn = lpips.LPIPS(net='alex').cuda()

def load_i3d_model():
    model = r3d_18(pretrained=True)
    model.fc = nn.Identity() 
    return model.cuda().eval()

i3d_model = load_i3d_model()

def calculate_fvd(preds, targets):
    B, L, _, H, W = preds.shape
    preds_rgb = preds.repeat(1, 1, 3, 1, 1).view(B, L, 3, H, W).permute(0, 2, 1, 3, 4).cuda()
    targets_rgb = targets.repeat(1, 1, 3, 1, 1).view(B, L, 3, H, W).permute(0, 2, 1, 3, 4).cuda()
    with torch.no_grad():
        pred_features = i3d_model(preds_rgb).view(B, -1)
        target_features = i3d_model(targets_rgb).view(B, -1)
    mu1, sigma1 = pred_features.mean(dim=0), torch.cov(pred_features.T)
    mu2, sigma2 = target_features.mean(dim=0), torch.cov(target_features.T)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy(), disp=False)
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + 1e-6 * torch.eye(sigma1.shape[0])).cpu().numpy() @ (sigma2 + 1e-6 * torch.eye(sigma2.shape[0])).cpu().numpy())
    covmean = torch.tensor(covmean).float().cuda()
    fvd = diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    return fvd.item()

def compute_metrics(preds, targets):
    preds = preds.cpu()
    targets = targets.cpu()
    ssim_vals = []
    psnr_vals = []
    lpips_vals = []
    for b in range(preds.shape[0]):
        pred = preds[b]
        target = targets[b]
        for frame in range(pred.shape[0]):
            pred_frame_np = pred[frame].squeeze().numpy()
            target_frame_np = target[frame].squeeze().numpy()
            ssim_vals.append(ssim(target_frame_np, pred_frame_np, data_range=1.0))
            psnr_vals.append(psnr(target_frame_np, pred_frame_np, data_range=1.0))
            lpips_val = lpips_fn(torch.tensor(target[frame]).expand(1, 3, pred.shape[2], pred.shape[3]).squeeze().cuda(), torch.tensor(pred[frame]).expand(1, 3, pred.shape[2], pred.shape[3]).squeeze().cuda())
            lpips_vals.append(lpips_val.item())
    
    fvd_val = calculate_fvd(preds, targets)
    return np.mean(ssim_vals), np.mean(psnr_vals), np.mean(lpips_vals), fvd_val

def visualize(GTs, PREDs, step, vis_num, sample_idx):
    L = GTs.size(1)
    indices = torch.linspace(0, L - 1, steps=vis_num).long()
    GTs, PREDs = GTs[:, indices], PREDs[:, indices]
    _, axes = plt.subplots(2, vis_num, figsize=(20, 5))
    for i in range(vis_num):
        axes[0, i].imshow(GTs[0, i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(f"GT {indices[i].item()}")
        axes[0, i].axis('off')
        axes[1, i].imshow(PREDs[0, i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1, i].set_title(f"Pred {indices[i].item()}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.vis_path, f'out_{step}_sample_{sample_idx}.png'))
    plt.close()

model.eval()
running_loss = 0.0
ssim_vals, psnr_vals, lpips_vals, fvd_vals = [], [], [], []
with torch.no_grad():
    for step, (inputs, outputs) in enumerate(tqdm(dataloader)):
        inputs, outputs = inputs.cuda(), outputs.cuda()
        pred_outputs = model(inputs, mode='i_sigmoid', out_frames_num=args.eval_n_frames_output)
        loss = loss_fn(pred_outputs, outputs)
        running_loss += loss.item()
        logging.info(f'Step {step+1}, Average Loss: {running_loss}')
        tqdm.write(f'Step {step+1}, Average Loss: {running_loss}')
        
        ssim_val, psnr_val, lpips_val, fvd_val = compute_metrics(pred_outputs, outputs)
        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)
        lpips_vals.append(lpips_val)
        fvd_vals.append(fvd_val)
        
        logging.info(f'Mean SSIM: {np.mean(ssim_vals)}, Mean PSNR: {np.mean(psnr_vals)}, Mean LPIPS: {np.mean(lpips_vals)}, Mean FVD: {np.mean(fvd_vals)}')
        tqdm.write(f'Mean SSIM: {np.mean(ssim_vals)}, Mean PSNR: {np.mean(psnr_vals)}, Mean LPIPS: {np.mean(lpips_vals)}, Mean FVD: {np.mean(fvd_vals)}')
        
        running_loss = 0.0
        for sample_idx in range(outputs.size(0)):
            visualize(outputs[sample_idx:sample_idx+1], pred_outputs[sample_idx:sample_idx+1], step + 1, args.eval_vis_num, sample_idx)

logging.shutdown()
