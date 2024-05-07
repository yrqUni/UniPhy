import torch
from ConvLRU import ConvLRU

class Args:
    input_size = 100
    hidden_ch = 8
    emb_ch = 4 
    input_ch = 3
    convlru_dropout = 0.1  
    ffn_dropout = 0.1
    convlru_num_blocks = 12
args = Args()
model = ConvLRU(args)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt.zero_grad()
input = torch.randn(8, 32, 3, 100, 100)
label = torch.randn(8, 32, 3, 100, 100)
outputs = model(input)
loss = loss_fn(outputs, label)
loss.backward()
opt.step()
print(f"Done! Loss {loss}")
