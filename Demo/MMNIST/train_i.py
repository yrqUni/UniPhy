import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Model'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from ModelConvLRU import ConvLRU 
from DATA.MMNIST import MovingMNIST 

class Args:
    # input info
    input_size = (64, 64)
    input_ch = 1
    # convlru info
    emb_ch = 12
    convlru_dropout = 0.0
    convlru_num_blocks = 6
    #
    hidden_factor = (1, 1)
    use_resnet = False
    resnet_type = 'resnet18'
    resnet_path = './resnet_ckpt'
    resnet_pretrained = True
    resnet_trainable = True
    resnet_scale_factor = 8
    # emb info
    emb_hidden_ch = 64
    emb_dropout = 0.0
    emb_hidden_layers_num = 4
    # ffn info
    ffn_hidden_ch = 32
    ffn_dropout = 0.0
    ffn_hidden_layers_num = 2
    # dec info
    dec_hidden_ch = 64
    dec_dropout = 0.0
    dec_hidden_layers_num = 4
    # data info
    root = './DATA/MMNIST/'
    is_train = True
    n_frames_input = 8
    n_frames_output = 8
    num_objects = [2]
    num_samples = int(5e3)
    # training info
    batch_size = 16
    lr = 1e-3
    EPs = 500
    vis = 50
    out_path = './exp2/'
    log_file = os.path.join(out_path, 'log')
    ckpt_path = os.path.join(out_path, 'ckpt/')
    vis_path = os.path.join(out_path, 'vis/')
    pretrain_path = '230_250.pth'
    def __str__(self):
        attrs = vars(self)
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())
args = Args()

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
if not os.path.exists(args.ckpt_path):
    os.makedirs(args.ckpt_path)
if not os.path.exists(args.vis_path):
    os.makedirs(args.vis_path)

logging.basicConfig(filename=args.log_file, level=logging.INFO)
logging.info(logging.info(args.__str__()))

dataset = MovingMNIST(root=args.root, is_train=args.is_train, n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output, num_objects=args.num_objects, num_samples=args.num_samples)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

model = ConvLRU(args).cuda()

if os.path.exists(args.pretrain_path):
    model.load_state_dict(torch.load(args.pretrain_path))
    logging.info(f'Loaded pretrained model from {args.pretrain_path}')
else:
    logging.info('No pretrained model found, starting from scratch.')

loss_fn = nn.BCEWithLogitsLoss().cuda()
opt = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataloader))

def visualize(GTs, PREDs, epoch, step):
    L = GTs.size(1)
    GTs, PREDs = GTs.squeeze(2), PREDs.squeeze(2)
    _, axes = plt.subplots(2, L, figsize=(20, 5))
    for i in range(L):
        axes[0, i].imshow(GTs[0, i].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"GT {i}")
        axes[0, i].axis('off')
        axes[1, i].imshow(torch.sigmoid(PREDs[0, i]).cpu().detach().numpy(), cmap='gray')
        axes[1, i].set_title(f"Pred {i}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.vis_path, f'visualization_epoch{epoch}_step{step}.png'))
    plt.close()

for ep in range(args.EPs):
    model.train()
    running_loss = 0.0
    for step, (inputs, outputs) in enumerate(tqdm(dataloader)):
        inputs, outputs = inputs.cuda(), outputs.cuda()
        opt.zero_grad()
        pred_outputs = model(inputs, mode='i', out_frames_num=args.n_frames_output)
        # pred_outputs = torch.sigmoid(pred_outputs) # if BCEWithLogitsLoss, no need to sigmoid for pred_outputs 
        loss = loss_fn(pred_outputs, outputs)
        loss.backward()
        opt.step()
        scheduler.step()
        running_loss += loss.item()
        if (step + 1) % args.vis == 0:
            avg_loss = running_loss / args.vis
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, LR: {current_lr}')
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, f'{ep}_{step+1}.pth'))
            tqdm.write(f'Step {step+1}, Epoch {ep}, Average Loss: {avg_loss}, LR: {current_lr}')
            running_loss = 0.0
            visualize(outputs, pred_outputs, ep, step + 1)

logging.shutdown()
