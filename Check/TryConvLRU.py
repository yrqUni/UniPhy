import sys
sys.path.append('../Model/ConvLRU/')

import gc
import torch
from ModelConvLRU import ConvLRU

class Args:
    def __init__(self):
        # sample H, W
        self.sample_input_size = (720, 1440)
        # input info
        self.input_size = (720, 1440)
        self.input_ch = 20
        self.out_ch = 20
        # convlru info
        self.emb_ch = 32
        self.convlru_num_blocks = 1
        #
        self.hidden_factor = (10, 20)
        self.use_mhsa = True
        self.use_gate = True
        # emb info
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.emb_strategy = 'pxus'
        # ffn info
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        # dec info
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.dec_strategy = 'pxsf'
        # output info
        self.gen_factor = 8
        # activation
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Sigmoid'
args = Args()
B = 2
L = 8

model = ConvLRU(args) # .to('mps')
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GP: {total_params}")
print(f"TP: {trainable_params}")
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
inputs_train = torch.randn(B, L, args.input_ch, *args.sample_input_size) # .to('mps')
labels_train = torch.randn(B, L, args.input_ch, *args.sample_input_size) # .to('mps')
opt.zero_grad()
outputs = model(inputs_train, mode='p')
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"\nSuccessful! p mode Loss {loss}\n")
del model, loss_fn, opt, inputs_train, labels_train, outputs
torch.cuda.empty_cache()
gc.collect()

out_frames_num = 8
model = ConvLRU(args) # .to('mps')
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GP: {total_params}")
print(f"TP: {trainable_params}")
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
inputs_train = torch.randn(B, L, args.input_ch, *args.sample_input_size) # .to('mps')
labels_train = torch.randn(B, out_frames_num, args.input_ch, *args.sample_input_size) # .to('mps')
opt.zero_grad()
out_gen_num = out_frames_num // args.gen_factor
outputs = model(inputs_train, mode='i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"\nSuccessful! i mode Loss {loss}\n")
