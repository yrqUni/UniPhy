import torch
from ConvLRU import IterativeConvLRU

out_frames = 8
class Args:
    input_size = 100
    input_ch = 3
    hidden_ch = 8
    emb_ch = 4 
    convlru_dropout = 0.1  
    ffn_dropout = 0.1
    convlru_num_blocks = 12
args = Args()
model = IterativeConvLRU(args).cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt.zero_grad()
inputs = torch.randn(2, 8, args.input_ch, args.input_size, args.input_size).cuda()
labels = torch.randn(2, out_frames, args.input_ch, args.input_size, args.input_size).cuda()
outputs = model(inputs, out_frames)
print(outputs.shape)
loss = loss_fn(outputs, labels)
loss.backward()
opt.step()
print(f"Done! Loss {loss}")
