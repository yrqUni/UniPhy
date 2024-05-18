import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import torch
from Model.ConvLRU import ConvLRU

class Args:
    # input info
    input_size = (300, 200)
    input_ch = 1
    # convlru info
    emb_ch = 4
    convlru_dropout = 0.1  
    convlru_num_blocks = 12
    #
    hidden_factor = (3, 2)
    # emb info
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
inputs_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
labels_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
opt.zero_grad()
outputs, hiddens = model(inputs_train, mode='p')
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"\nSuccessful! p mode Loss {loss}\n")
del model, loss_fn, opt, inputs_train, labels_train, outputs, hiddens
torch.cuda.empty_cache()
gc.collect()

out_frames_num = 8
model = ConvLRU(args).cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
inputs_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
labels_train = torch.randn(B, out_frames_num, args.input_ch, *args.input_size).cuda()
opt.zero_grad()
outputs = model(inputs_train, mode='i', out_frames_num=out_frames_num)
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"\nSuccessful! i mode Loss {loss}\n")
