import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.ConvLRU import ConvLRU

class Args:
    # input info
    input_size = 32
    input_ch = 1
    # convlru info
    convlru_hidden_ch = 8
    convlru_dropout = 0.1  
    convlru_num_blocks = 12
    #
    hidden_factor = 2
    # emb info
    emb_ch = 4
    emb_hidden_ch = 8
    emb_dropout = 0.0
    emb_hidden_layers_num = 1
    # ffn info
    ffn_hidden_ch = 32
    ffn_dropout = 0.1
    ffn_hidden_layers_num = 4
    # dec info
    dec_hidden_ch = 32
    dec_dropout = 0.1
    dec_hidden_layers_num = 8
args = Args()
B = 2
L = 8
model = ConvLRU(args).cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
inputs_train = torch.randn(B, L, args.input_ch, args.input_size, args.input_size).cuda()
labels_train = torch.randn(B, L, args.input_ch, args.input_size, args.input_size).cuda()
opt.zero_grad()
outputs = model(inputs_train, mask=None, mode='train', out_frames=None)
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"Train mode Loss {loss}")
out_frames = 4
inputs_train = torch.randn(B, L, args.input_ch, args.input_size, args.input_size).cuda()
mask = (torch.randn(B, L).cuda() > 0).float() 
labels_train = torch.randn(B, out_frames, args.input_ch, args.input_size, args.input_size).cuda()
opt.zero_grad()
outputs = model(inputs_train, mode='infer', out_frames=out_frames)
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"Infer mode Loss {loss}")
