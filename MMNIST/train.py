import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from Model.ConvLRU import ConvLRU 
from Data.MMNIST import MovingMNIST 

class Args:
    # input info
    input_size = (64, 64)
    input_ch = 1
    # convlru info
    emb_ch = 4
    convlru_dropout = 0.1  
    convlru_num_blocks = 3
    #
    hidden_factor = (1, 1)
    # emb info
    emb_hidden_ch = 64
    emb_dropout = 0.0
    emb_hidden_layers_num = 4
    # ffn info
    ffn_hidden_ch = 32
    ffn_dropout = 0.1
    ffn_hidden_layers_num = 2
    # dec info
    dec_hidden_ch = 64
    dec_dropout = 0.1
    dec_hidden_layers_num = 4
    # data info
    root = '../Data/MMNIST/'
    is_train = True
    n_frames_input = 17
    n_frames_output = 1
    num_objects = [3]
    # training info
    batch_size = 16
    lr = 1e-4
    num_steps = 400000
    log_file = f'log_{convlru_num_blocks}'
    save_path = f'ckpt_{convlru_num_blocks}/'
args = Args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

logging.basicConfig(filename=args.log_file, level=logging.INFO)

dataset = MovingMNIST(root=args.root, is_train=args.is_train, n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output, num_objects=args.num_objects, num_samples=int(1e10))
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
data_iter = iter(dataloader)

model = ConvLRU(args).cuda()

loss_fn = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_steps)

for step in tqdm(range(args.num_steps)):
    inputs, _ = next(data_iter)
    inputs = inputs.cuda()
    opt.zero_grad()
    pred_outputs = model(inputs[:, :-1], mode='p')
    loss = loss_fn(pred_outputs, inputs[:, 1:])
    loss.backward()
    opt.step()
    scheduler.step()
    if step % 100 == 0:
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f'Step {step}, Loss: {loss.item()}, LR: {current_lr}')
        torch.save(model.state_dict(), os.path.join(args.save_path, f'{step}.pth'))
    tqdm.write(f'Step {step}, Loss: {loss.item()}, LR: {current_lr}')
logging.shutdown()
