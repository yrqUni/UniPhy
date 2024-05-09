import torch
from ConvLRU import OnlyIterativeInfer_ConvLRU

class Args:
    input_size = 100
    input_ch = 3
    hidden_ch = 8
    emb_ch = 4 
    convlru_dropout = 0.1  
    ffn_dropout = 0.1
    convlru_num_blocks = 12
    use_resnet = True
    resnet_type = 'resnet34' # resnet18, resnet34, resnet50, resnet101, resnet152
    resnet_pretrained = True
    resnet_trainable = True
args = Args()
model = OnlyIterativeInfer_ConvLRU(args).cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt.zero_grad()
inputs_train = torch.randn(2, 8, args.input_ch, args.input_size, args.input_size).cuda()
labels_train = torch.randn(2, 8, args.input_ch, args.input_size, args.input_size).cuda()
outputs = model(inputs_train, mode='train', out_frames=None)
loss = loss_fn(outputs, labels_train)
loss.backward()
opt.step()
print(f"Loss {loss}")
out_frames = 4
inputs_train = torch.randn(2, 8, args.input_ch, args.input_size, args.input_size).cuda()
labels_train = torch.randn(2, 8, args.input_ch, args.input_size, args.input_size).cuda()
outputs = model(inputs_train, mode='infer', out_frames=out_frames)
print(f"Done! Inference {outputs.shape}")
