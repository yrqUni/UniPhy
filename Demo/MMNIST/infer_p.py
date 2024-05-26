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
        self.dec_attn_layers_num = 3
        self.dec_attn_dim = 768
        self.dec_attn_num_heads = 8
        self.dec_attn_ffn_dim_factor = 1
        self.dec_attn_dropout = 0.0
        self.dec_hidden_ch = 64
        self.dec_dropout = 0.0
        self.dec_hidden_layers_num = 4
        # data info
        self.root = './DATA/MMNIST/'
        self.is_train = True
        self.n_frames_input = 128
        self.n_frames_output = 128
        self.num_objects = [2]
        self.num_samples = int(5e3)
        # training info
        self.batch_size = 2
        self.lr = 1e-3
        self.EPs = 500
        self.vis_num = 16
        self.out_path = './exp_a/'
        self.log_file = os.path.join(self.out_path, 'log')
        self.vis_path = os.path.join(self.out_path, 'vis/')
        self.pretrain_path = '/data1/ruiqingy/Workspace/ConvLRU/Demo/MMNIST/exp1/ckpt/A.pth'
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
with torch.no_grad():
    for step, (inputs, outputs) in enumerate(tqdm(dataloader)):
        inputs, outputs = inputs.cuda(), outputs.cuda()
        out = []
        pred_outputs = model(inputs, mode='p_sigmoid')[:, -1:]
        out.append(pred_outputs)
        for i in range(args.n_frames_output-1):
            inputs = torch.cat([inputs[:, 1:], out[i]], dim=1)
            pred_outputs = model(inputs, mode='p_sigmoid')[:, -1:]
            out.append(pred_outputs)
        pred_outputs = torch.cat(out, dim=1)
        loss = loss_fn(pred_outputs, outputs)
        running_loss += loss.item()
        logging.info(f'Step {step+1}, Average Loss: {running_loss}')
        tqdm.write(f'Step {step+1}, Average Loss: {running_loss}')
        running_loss = 0.0
        for sample_idx in range(outputs.size(0)):
            visualize(outputs[sample_idx:sample_idx+1], pred_outputs[sample_idx:sample_idx+1], step + 1, args.vis_num, sample_idx)

logging.shutdown()
