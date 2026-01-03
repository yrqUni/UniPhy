import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model/ConvLRU')))
from ModelConvLRU import ConvLRU

class Args:
    def __init__(self):
        self.input_ch = 2
        self.out_ch = 2
        self.emb_ch = 16
        self.hidden_factor = (2, 2)
        self.convlru_num_blocks = 2
        self.lru_rank = 8
        self.input_size = (64, 64)
        self.head_mode = "gaussian"

def check():
    args = Args()
    model = ConvLRU(args).cuda()
    x = torch.randn(1, 4, 2, 64, 64).cuda()
    listT = torch.ones(1, 4).cuda()
    
    print("Checking Parallel Mode...")
    y = model(x, mode="p", listT=listT)
    print(f"Output shape: {y.shape}")
    
    print("Checking Autoregressive Mode...")
    y_ar = model(x[:, :1], mode="i", out_gen_num=4, listT=listT[:, :1], listT_future=listT[:, 1:])
    print(f"Inference shape: {y_ar.shape}")

if __name__ == "__main__":
    check()

