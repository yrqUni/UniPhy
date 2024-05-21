import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import torch
from Model.ModelConvLRU import ConvLRU

class Args:
    def __init__(self):
        # input info
        self.input_size = (64, 64)
        self.input_ch = 1
        # convlru info
        self.emb_ch = 4
        self.convlru_dropout = 0.1  
        self.convlru_num_blocks = 12
        #
        self.hidden_factor = (1, 1)
        self.use_resnet = True
        self.resnet_type = 'resnet18'
        self.resnet_path = './resnet_ckpt'
        self.resnet_pretrained = True
        self.resnet_trainable = True
        self.resnet_scale_factor = 8
        # emb info
        self.emb_hidden_ch = 8
        self.emb_dropout = 0.0
        self.emb_hidden_layers_num = 1
        # ffn info
        self.ffn_hidden_ch = 32
        self.ffn_dropout = 0.1
        self.ffn_hidden_layers_num = 4
        # dec info
        self.dec_hidden_ch = 32
        self.dec_dropout = 0.1
        self.dec_hidden_layers_num = 8
args = Args()
B = 2
L = 8

model = ConvLRU(args).cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
inputs_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
labels_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
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
