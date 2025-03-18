import sys
import os
import gc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Git.ConvLRU.Model.FFT.ModelConvLRU import ConvLRU

class Args:
    def __init__(self):
        self.input_size = (32, 32)
        self.input_ch = 4
        self.emb_ch = 32
        self.convlru_num_blocks = 8
        self.hidden_factor = (4, 4)
        self.io_use_large_kernel = False
        self.use_mhsa = False
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 4
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 2
        self.dec_hidden_ch = 32
        self.dec_hidden_layers_num = 4
        self.gen_factor = 8
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Sigmoid'

def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def demo_ddp(rank, world_size, args):
    """DDP demo with ConvLRU model."""
    setup_ddp(rank, world_size)
    B = 8
    L = 8
    
    model = ConvLRU(args).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:  
        print(f"GP: {total_params}")
        print(f"TP: {trainable_params}")

    loss_fn = torch.nn.MSELoss().cuda(rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    inputs_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda(rank)
    labels_train = torch.randn(B, L, args.input_ch, *args.input_size).cuda(rank)

    dataset = TensorDataset(inputs_train, labels_train)
    train_loader = DataLoader(dataset, batch_size=1)

    model.train()
    for inputs, labels in train_loader:
        opt.zero_grad()
        outputs = model(inputs, mode='p')
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
    
    if rank == 0:  
        print(f"\nSuccessful! p mode Loss {loss}\n")

    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(demo_ddp,
                                    args=(world_size, args,),
                                    nprocs=world_size,
                                    join=True)
    else:
        print("This script requires multiple GPUs to run.")
