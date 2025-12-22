import torch
import torch.nn as nn
import sys
import os

# 确保能导入 ModelConvLRU
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Model", "ConvLRU")))

try:
    from ModelConvLRU import ConvLRU, Embedding, ConvLRUModel, ConvLRULayer, Decoder
except ImportError:
    # 假设脚本放在 Check 目录下，尝试直接导入
    sys.path.append("/nfs/ConvLRU/Model/ConvLRU")
    from ModelConvLRU import ConvLRU, Embedding, ConvLRUModel, ConvLRULayer, Decoder

class MockArgs:
    def __init__(self):
        self.input_ch = 4
        self.input_size = (32, 32)
        self.emb_ch = 16
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.static_ch = 2
        self.hidden_factor = (2, 2)
        self.unet = False
        self.convlru_num_blocks = 1
        self.ffn_hidden_layers_num = 1
        self.use_cbam = False
        self.num_expert = -1
        self.activate_expert = 2
        self.lru_rank = 8
        self.use_selective = False
        self.bidirectional = False
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0
        self.head_mode = "gaussian"
        self.out_ch = 4
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"
        self.use_checkpointing = False # 调试时关闭 checkpoint 以便看到中间层

def hook_fn(name):
    def hook(module, input, output):
        in_shape = input[0].shape if isinstance(input, tuple) else input.shape
        out_shape = output.shape if isinstance(output, torch.Tensor) else (output[0].shape if isinstance(output, tuple) else "Tuple/List")
        print(f"[\033[94m{name}\033[0m] In: {in_shape} -> Out: {out_shape}")
        if hasattr(module, 'emb_ch'):
             print(f"    Config: emb_ch={module.emb_ch}")
    return hook

def debug_architecture_test():
    print("\n=== Debugging Architecture Combinations (B=2, L=4) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    model = ConvLRU(args).to(device)
    
    # 注册 Hooks
    model.embedding.register_forward_hook(hook_fn("Embedding"))
    model.convlru_model.register_forward_hook(hook_fn("ConvLRUModel"))
    # Hook 第一个 Layer
    if model.convlru_model.convlru_blocks:
        model.convlru_model.convlru_blocks[0].lru_layer.register_forward_hook(hook_fn("Layer0.LRU"))
    
    B, L, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    
    print(f"Input x: {x.shape}")
    print(f"Input listT: {listT.shape}")
    
    try:
        out = model(x, mode="p", listT=listT, static_feats=static)
        print("Forward Pass Successful")
    except Exception as e:
        print(f"\033[91mError Caught:\033[0m {e}")
        import traceback
        traceback.print_exc()

def debug_consistency_test():
    print("\n=== Debugging Consistency Step-1 (B=1, L=1) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = MockArgs()
    # 强制复现 consistency test 的设置
    args.unet = True 
    args.num_expert = 4
    
    model = ConvLRU(args).to(device)
    
    model.embedding.register_forward_hook(hook_fn("Embedding"))
    if model.convlru_model.down_blocks:
        model.convlru_model.down_blocks[0].lru_layer.register_forward_hook(hook_fn("DownBlock0.LRU"))
    
    B, L, C, H, W = 1, 1, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device) # start_frame
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device) # listT[:, 0:1]
    
    print(f"Input x: {x.shape}")
    print(f"Input listT: {listT.shape}")
    
    try:
        out = model(x, mode="p", listT=listT, static_feats=static)
        print("Forward Pass Successful")
    except Exception as e:
        print(f"\033[91mError Caught:\033[0m {e}")
        # 不需要 full traceback 这里，主要看上面的 hook 输出

if __name__ == "__main__":
    debug_architecture_test()
    debug_consistency_test()