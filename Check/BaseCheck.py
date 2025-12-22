import os
import sys
import torch
import torch.nn as nn

# 路径设置，确保能导入模型
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU

# -----------------------------------------------------------------------------
# 1. 辅助工具：Hook 函数
# -----------------------------------------------------------------------------
def get_shape_str(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    elif isinstance(x, (tuple, list)):
        return f"[{', '.join([get_shape_str(i) for i in x])}]"
    return str(type(x))

def register_hooks(model):
    print("\n>>> Registering Probe Hooks...")
    
    # Hook 1: 监控 Decoder 内部最终输出前的形状
    def decoder_output_hook(module, input, output):
        # input is tuple, output is Tensor or Tuple
        print(f"[\033[96mDecoder Output\033[0m] Shape: {get_shape_str(output)}")
        # 检查 Decoder 是否已经做过 Permute
        # 预期 Decoder 输出应该是 (B, L, C, H, W)
    
    model.decoder.register_forward_hook(decoder_output_hook)

    # Hook 2: 监控 RevIN 的行为
    def revin_hook(module, input, output):
        x = input[0]
        mode = input[1] if len(input) > 1 else "unknown"
        print(f"[\033[93mRevIN Input\033[0m]  Shape: {get_shape_str(x)} | Mode: {mode}")
        if mode == "denorm":
            print(f"    -> RevIN Stats: Mean={module.mean.shape}, Stdev={module.stdev.shape}")
            if x.size(2) != module.num_features: # 假设 dim 2 是 channel
                 print(f"    \033[91m[WARNING]\033[0m Tensor Channel Dim ({x.size(2)}) != RevIN Features ({module.num_features})")
    
    # 临时 Monkey Patch RevIN 的 forward 以捕获参数
    orig_revin_forward = model.revin.forward
    def hooked_revin_forward(x, mode):
        revin_hook(model.revin, (x, mode), None)
        return orig_revin_forward(x, mode)
    model.revin.forward = hooked_revin_forward

# -----------------------------------------------------------------------------
# 2. 模拟参数
# -----------------------------------------------------------------------------
class MockArgs:
    def __init__(self):
        self.input_ch = 30  # 输入通道
        self.out_ch = 9     # 输出通道 (不匹配，触发 Skip RevIN 逻辑)
        self.head_mode = "diffusion" # 只有 Diffusion/Regression 模式才会有这个问题
        
        # 其他必须参数
        self.input_size = (32, 32)
        self.emb_ch = 16
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 0
        self.static_ch = 0
        self.hidden_factor = (1, 1)
        self.unet = False
        self.convlru_num_blocks = 1
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 0
        self.use_cbam = False
        self.num_expert = -1
        self.activate_expert = 2
        self.lru_rank = 8
        self.use_selective = False
        self.bidirectional = False
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"
        self.use_checkpointing = False

# -----------------------------------------------------------------------------
# 3. 执行诊断
# -----------------------------------------------------------------------------
def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Diagnostic on: {device}")
    
    args = MockArgs()
    model = ConvLRU(args).to(device)
    model.eval()
    
    register_hooks(model)

    B, L, H, W = 1, 2, 32, 32
    # 输入维度: (B, L, C_in, H, W)
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    
    print("\n>>> Starting Forward Pass...")
    try:
        out = model(x, mode="p", listT=None)
        print(f"\n[\033[92mFinal Output\033[0m]   Shape: {tuple(out.shape)}")
        
        # 验证
        expected = (B, L, args.out_ch, H, W)
        if tuple(out.shape) == expected:
            print("\n✅ Shape Matches Expectation.")
        else:
            print(f"\n❌ Shape Mismatch! Expected {expected}, Got {tuple(out.shape)}")
            print("Analyze: Did dimensions L(2) and C(9) get swapped?")
            
    except Exception as e:
        print(f"\n\033[91m[CRASH]\033[0m {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostic()
