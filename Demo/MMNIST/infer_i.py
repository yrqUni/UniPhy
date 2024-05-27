import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Model'))

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
    def __init__(self):
        # input info
        self.input_size = (64, 64)
        self.input_ch = 1
        # convlru info
        self.emb_ch = 128
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
        self.dec_attn_layers_num = 1
        self.dec_attn_ch = 1
        self.dec_attn_num_heads = 8
        self.dec_attn_ffn_dim_factor = 1
        self.dec_attn_dropout = 0.0
        self.dec_hidden_ch = 64
        self.dec_dropout = 0.0
        self.dec_hidden_layers_num = 4
        # data info
        self.root = './DATA/MMNIST/'
        self.is_train = True
        self.n_frames_input = 8
        self.n_frames_output = 32
        self.num_objects = [2]
        self.num_samples = int(5e3)
        # training info
        self.batch_size = 2
        self.lr = 1e-3
        self.EPs = 500
        self.vis_num = 16
        self.out_path = './exp_b/'
        self.log_file = os.path.join(self.out_path, 'log')
        self.vis_path = os.path.join(self.out_path, 'vis/')
        self.pretrain_path = '/data1/ruiqingy/Workspace/ConvLRU/Demo/MMNIST/exp_mix_0/ckpt/A.pth'
    def __str__(self):
        attrs = vars(self)
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())
args = Args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
if not os.path.exists(args.vis_path):
    os.makedirs(args.vis_path)

logging.basicConfig(filename=args.log_file, level=logging.INFO)
logging.info(str(args))
print(str(args))

dataset = MovingMNIST(root=args.root, is_train=args.is_train, n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output, num_objects=args.num_objects, num_samples=args.num_samples)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

model = ConvLRU(args).cuda()

if os.path.exists(args.pretrain_path):
    model.load_state_dict(torch.load(args.pretrain_path))
    logging.info(f'Loaded pretrained model from {args.pretrain_path}')
else:
    logging.info('No pretrained model found, starting from scratch.')

loss_fn = nn.BCELoss().cuda()

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
        pred_outputs = model(inputs, mode='i_sigmoid', out_frames_num=args.n_frames_output)
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
            visualize(outputs[sample_idx:sample_idx+1], pred_outputs[sample_idx:sample_idx+1], step + 1, args.vis_num, sample_idx)

logging.shutdown()