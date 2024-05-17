import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Model.ConvLRU import ConvLRU, ConvLRULayer

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
outputs = model(x = inputs_train, out_frames = None, mode = 'p')
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"p mode Loss {loss}")
out_frames = 4
inputs_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda()
labels_train = torch.randn(B, out_frames, args.input_ch, *args.input_size).cuda()
opt.zero_grad()
outputs = model(x = inputs_train, out_frames = out_frames, mode = 'i')
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"i mode Loss {loss}")

# with torch.no_grad():
#     class Args:
#         emb_ch = 8
#         convlru_dropout = 0.1
#     args = Args()
#     convlru_layer = ConvLRULayer(args, input_downsp_shape=(args.emb_ch, 16, 16))
#     torch.manual_seed(0)
#     convlru_layer.eval()
#     B, L, C, H, W = 2, 4, args.emb_ch, 16, 16
#     x = torch.randn(B, L, C, H, W)
#     mask = torch.ones(B, L)
#     x_parallel_hidden = convlru_layer.convlru_parallel_mode(x, mask)
#     x_hidden = convlru_layer.convlru_iter_mode(x[:, -1, :, :, :].unsqueeze(1) , x_parallel_hidden[:, -2, :, :, :].unsqueeze(1))
#     # print((x_hidden - x_parallel_hidden[:, -1, :, :, :].unsqueeze(1)).abs().max())
