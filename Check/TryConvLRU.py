import os
import sys
import math
import torch
import numpy as np

# Adjust path to find the Model
sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))
from ModelConvLRU import ConvLRU
from pscan import pscan_check

# Determinism settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(s=0):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArgsBase:
    def __init__(self):
        self.input_size = (64, 64)
        self.input_ch = 4
        self.out_ch = 4
        self.emb_ch = 16
        self.emb_hidden_ch = 16
        self.emb_hidden_layers_num = 1
        self.emb_strategy = "pxus"
        self.hidden_factor = (2, 2)
        self.convlru_num_blocks = 2
        self.lru_rank = 16
        self.use_gate = True
        self.use_cbam = False
        self.use_freq_prior = False
        self.freq_rank = 8
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.ffn_hidden_ch = 16
        self.ffn_hidden_layers_num = 1
        self.hidden_activation = "SiLU"
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        # [NEW] Static features channel count
        self.static_ch = 0

def make_test_configs():
    cfgs = []
    
    # Case 1: Standard Base
    a = ArgsBase()
    cfgs.append(("base_pxsf_gate", a))
    
    # Case 2: Physics Priors (FNO + SH)
    a = ArgsBase()
    a.use_freq_prior = True
    a.use_sh_prior = True
    cfgs.append(("with_physics_priors", a))
    
    # Case 3: Static Features (FiLM)
    a = ArgsBase()
    a.static_ch = 3 # e.g., LSM, Orography, Lat/Lon
    cfgs.append(("with_static_features", a))
    
    # Case 4: Conv Downsample + Deconv Upsample
    a = ArgsBase()
    a.emb_strategy = "conv"
    a.dec_strategy = "deconv"
    a.dec_hidden_layers_num = 1
    cfgs.append(("conv_io_deconv", a))
    
    # Case 5: Minimal (No Gate, No CBAM)
    a = ArgsBase()
    a.use_gate = False
    a.use_cbam = False
    cfgs.append(("minimal_no_gate", a))
    
    # Case 6: CBAM Attention
    a = ArgsBase()
    a.use_cbam = True
    cfgs.append(("with_cbam", a))
    
    return cfgs

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def forward_full_p(model, x, listT=None, static_feats=None):
    return model(x, mode="p", listT=listT, static_feats=static_feats)

@torch.no_grad()
def forward_streaming_p_equiv(model, x, chunk_sizes, listT=None, static_feats=None):
    """
    Simulates streaming by chunking the time dimension.
    Manually calls embedding -> core -> decoder to verify state passing.
    """
    B, L, C, H, W = x.shape
    em = model.embedding
    dm = model.decoder
    m = model.convlru_model
    outs = []
    last_hidden_list = None
    pos = 0
    
    for n in chunk_sizes:
        if pos >= L:
            break
        n = min(n, L - pos)
        x_chunk = x[:, pos : pos + n]
        
        # [UPDATE] Embedding now returns (x, cond) due to FiLM
        xe, cond = em(x_chunk, static_feats=static_feats)
        
        listT_slice = listT[:, pos : pos + n] if listT is not None else None
        
        # [UPDATE] Core model accepts cond
        xe, last_hidden_list = m(xe, last_hidden_ins=last_hidden_list, listT=listT_slice, cond=cond)
        
        # [UPDATE] Decoder accepts cond
        yo = dm(xe, cond=cond)
        
        outs.append(yo)
        pos += n
        
    return torch.cat(outs, dim=1)

def check_autoregressive_generation(model, x, listT, static_feats=None, out_gen_num=4):
    """
    Tests the Autoregressive/Inference branch (mode != 'p').
    This ensures the 'else' block in ConvLRU.forward works correctly.
    """
    print(f"   [test] Autoregressive Generation (Gen {out_gen_num} frames)...")
    model.eval()
    
    B, L, C, H, W = x.shape
    # We use the full 'x' as context, and predict 'out_gen_num' future frames
    
    # Fake future time intervals
    listT_future = torch.ones(B, out_gen_num, device=x.device, dtype=x.dtype)
    
    try:
        y_gen = model(x, mode="i", out_gen_num=out_gen_num, listT=listT, listT_future=listT_future, static_feats=static_feats)
    except Exception as e:
        print(f"     ❌ Autoregressive forward failed!")
        raise e

    # Check Output Shape
    # Expected: [B, out_gen_num, out_ch*2, H, W]
    # Note: ConvLRU.forward logic accumulates [next_step, next_step...]. 
    # Usually it returns `out_gen_num` frames.
    
    expected_shape = (B, out_gen_num, model.args.out_ch * 2, H, W)
    if y_gen.shape != expected_shape:
        print(f"     ❌ Gen Shape mismatch! Expected {expected_shape}, got {y_gen.shape}")
        return False
        
    print(f"     ✅ OK. Output shape: {y_gen.shape}")
    return True

def max_err(a, b):
    return float((a - b).abs().max().detach().cpu())

def check_output_shape(y, B, L, out_ch, H, W):
    expected_shape = (B, L, out_ch * 2, H, W)
    if y.shape != expected_shape:
        print(f"❌ Output shape mismatch! Expected {expected_shape}, got {y.shape}")
        return False
    return True

def check_sde_noise(model, x, listT, static_feats=None):
    print("   [test] SDE Noise Injection...")
    model.train()
    y1 = model(x, mode="p", listT=listT, static_feats=static_feats)
    y2 = model(x, mode="p", listT=listT, static_feats=static_feats)
    diff_train = max_err(y1, y2)
    
    model.eval()
    y3 = model(x, mode="p", listT=listT, static_feats=static_feats)
    y4 = model(x, mode="p", listT=listT, static_feats=static_feats)
    diff_eval = max_err(y3, y4)
    
    is_train_noisy = diff_train > 1e-6
    is_eval_det = diff_eval < 1e-9
    
    status = "OK" if (is_train_noisy and is_eval_det) else "FAIL"
    print(f"     -> Train Diff: {diff_train:.2e} (Expected > 0) | Eval Diff: {diff_eval:.2e} (Expected 0) => {status}")
    
    if not is_train_noisy:
        print("     ⚠️ Warning: Training mode seems deterministic. SDE noise might be too small or disabled.")
    if not is_eval_det:
        print("     ❌ Error: Eval mode is not deterministic!")
        return False
    return True

def expected_unused_name(name: str, args) -> bool:
    if not args.use_freq_prior and "freq_prior" in name:
        return True
    if not args.use_sh_prior and "sh_prior" in name:
        return True
    if not args.use_gate and "gate_conv" in name:
        return True
    if not args.use_cbam and "cbam" in name:
        return True
    # If static_ch is 0, static_embed params should not exist or be unused
    # Our Model code initializes them conditionally, so they shouldn't exist if ch=0.
    return False

def list_unused_parameters(model, x, listT, static_feats=None):
    model.train()
    model.zero_grad()
    y = model(x, mode="p", listT=listT, static_feats=static_feats)
    loss = y.sum()
    loss.backward()
    unused = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            unused.append(n)
    return unused

def run_test_case(name, args, device):
    print(f"\n>> Running Test Case: {name}")
    model = ConvLRU(args).to(device)
    total, trainable = count_params(model)
    print(f"   [params] Total: {total:,} | Trainable: {trainable:,}")
    
    B, L = 1, 8
    H, W = args.input_size
    dtype = torch.float32
    set_seed(42)
    
    x = torch.randn(B, L, args.input_ch, H, W, device=device, dtype=dtype)
    
    # [NEW] Generate static features if needed
    static_feats = None
    if args.static_ch > 0:
        static_feats = torch.randn(B, args.static_ch, H, W, device=device, dtype=dtype)
        print(f"   [input] Using Static Features: {static_feats.shape}")
        
    listT_ones = torch.ones(B, L, device=device, dtype=dtype)
    listT_rand = torch.rand(B, L, device=device, dtype=dtype) * 1.5 + 0.1
    
    model.eval()
    
    # 1. Parallel Mode Shape Check
    y = model(x, mode="p", listT=listT_ones, static_feats=static_feats)
    if not check_output_shape(y, B, L, args.out_ch, H, W):
        raise RuntimeError("Output Shape Check Failed")
        
    # 2. SDE Noise Check
    if not check_sde_noise(model, x, listT_rand, static_feats=static_feats):
        raise RuntimeError("SDE Noise Check Failed")
        
    # 3. Streaming Equivalence Check (Internal State Passing)
    print("   [test] Streaming Equivalence (P-Mode vs Chunked)...")
    patterns = [
        [1] * L,
        [L // 2, L - (L // 2)],
        [L]
    ]
    TOLERANCE = 1e-4
    for pat in patterns:
        model.eval()
        y_full = forward_full_p(model, x, listT=listT_rand, static_feats=static_feats)
        y_stream = forward_streaming_p_equiv(model, x, pat, listT=listT_rand, static_feats=static_feats)
        err = max_err(y_full, y_stream)
        
        if err > TOLERANCE:
            print(f"     ❌ FAIL pattern {pat}: Max Err {err:.2e} > {TOLERANCE}")
            raise RuntimeError("Streaming Equivalence Failed")
        else:
            print(f"     ✅ OK pattern {pat}: Max Err {err:.2e}")

    # 4. Autoregressive Generation Check (New)
    # This specifically checks the 'else' branch in forward(), loops, and listT_future
    if not check_autoregressive_generation(model, x, listT_rand, static_feats=static_feats, out_gen_num=4):
        raise RuntimeError("Autoregressive Generation Failed")
            
    # 5. Unused Params Check (Gradient Check)
    print("   [test] Unused Parameters (Gradient Check)...")
    unused = list_unused_parameters(model, x, listT=listT_rand, static_feats=static_feats)
    unexpected = [n for n in unused if not expected_unused_name(n, args)]
    
    if unexpected:
        print(f"     ❌ UNEXPECTED UNUSED PARAMS: {unexpected}")
        raise RuntimeError("Found unexpected unused parameters")
    else:
        print(f"     ✅ OK. (Ignored expected: {len(unused) - len(unexpected)})")

def main():
    print("========================================")
    print("      ConvLRU (Modern Arch) Test        ")
    print("========================================")
    
    print("[1/3] Checking PScan Operator...")
    if not pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
        print("❌ PScan Check Failed!")
        sys.exit(1)
    print("✅ PScan Check Passed.")
    
    device = pick_device()
    print(f"[2/3] Running on device: {device}")
    
    print("[3/3] Running Model Test Cases...")
    cfgs = make_test_configs()
    passed = 0
    
    for name, args in cfgs:
        try:
            run_test_case(name, args, device)
            passed += 1
        except Exception as e:
            print(f"\n❌ CASE FAILED: {name}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print(f"\n🎉 All {passed} test cases passed successfully!")

if __name__ == "__main__":
    main()
