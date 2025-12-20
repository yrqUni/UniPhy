import os
import sys
import torch

# Path Setup
sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))
from ModelConvLRU import ConvLRU
from pscan import pscan_check

# Determinism
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class MockArgs:
    def __init__(self):
        self.input_size = (32, 32)
        self.input_ch = 4
        self.out_ch = 4
        self.static_ch = 2
        
        self.emb_ch = 16
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.emb_strategy = "pxus"
        self.hidden_factor = (2, 2)
        
        self.convlru_num_blocks = 2
        self.lru_rank = 8
        
        self.use_gate = True
        self.use_cbam = False
        
        self.use_freq_prior = False
        self.freq_rank = 4
        self.use_sh_prior = False
        self.sh_Lmax = 4
        
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        
        # Advanced Features
        self.unet = False
        self.bidirectional = False
        self.use_selective = False
        self.head_mode = "gaussian"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_status(name, passed, msg=""):
    if passed:
        print(f"\033[92m[PASS] {name:<40}\033[0m {msg}")
    else:
        print(f"\033[91m[FAIL] {name:<40}\033[0m {msg}")

def test_configurations():
    print("\n=== 1. Architecture Combinations Test ===")
    device = get_device()
    
    configs = [
        ("Base_Flat", False, False, False),
        ("UNet_Only", True, False, False),
        ("BiDir_Only", False, True, False),
        ("Selective_Only", False, False, True),
        ("Full_Advanced", True, True, True),
    ]
    
    B, L, C, H, W = 2, 8, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    
    for name, unet, bidir, sel in configs:
        try:
            args = MockArgs()
            args.unet = unet
            args.bidirectional = bidir
            args.use_selective = sel
            
            model = ConvLRU(args).to(device)
            out = model(x, mode='p', listT=listT, static_feats=static)
            
            # Check Output Shape (Gaussian Head default: 2 * out_ch)
            expected_C = args.out_ch * 2
            if out.shape != (B, L, expected_C, H, W):
                raise ValueError(f"Shape Mismatch: {out.shape} vs {(B, L, expected_C, H, W)}")
            
            # Check Backward
            loss = out.sum()
            loss.backward()
            
            print_status(name, True, f"Params: {sum(p.numel() for p in model.parameters())}")
            
        except Exception as e:
            print_status(name, False, str(e))

def test_heads():
    print("\n=== 2. Decoder Heads Test ===")
    device = get_device()
    B, L, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    
    modes = ["gaussian", "diffusion", "token"]
    
    for mode in modes:
        try:
            args = MockArgs()
            args.head_mode = mode
            model = ConvLRU(args).to(device)
            
            timestep = None
            if mode == "diffusion":
                timestep = torch.randint(0, 1000, (B,), device=device).float()
                
            out = model(x, mode='p', listT=None, static_feats=None, timestep=timestep)
            
            passed = False
            msg = ""
            
            if mode == "gaussian":
                # Expect [B, L, 2*C, H, W]
                if out.shape[2] == args.out_ch * 2:
                    passed = True
                    msg = f"Shape: {out.shape}"
                    
            elif mode == "diffusion":
                # Expect [B, L, C, H, W]
                if out.shape[2] == args.out_ch:
                    passed = True
                    msg = f"Shape: {out.shape}"
                    
            elif mode == "token":
                # Expect Tuple (Quantized, Loss, Indices)
                if isinstance(out, tuple) and len(out) == 3:
                    quant, vq_loss, idx = out
                    if quant.shape[2] == args.out_ch:
                        loss = quant.sum() + vq_loss
                        loss.backward()
                        passed = True
                        msg = f"VQ Shape: {quant.shape}, Loss: {vq_loss.item():.4f}"
            
            print_status(f"Head: {mode}", passed, msg)
            
        except Exception as e:
            print_status(f"Head: {mode}", False, str(e))
            import traceback
            traceback.print_exc()

def test_consistency():
    print("\n=== 3. Consistency (Parallel vs Inference) ===")
    device = get_device()
    args = MockArgs()
    args.unet = True 
    args.bidirectional = False 
    model = ConvLRU(args).to(device)
    model.eval()
    
    B, L, C, H, W = 1, 6, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    
    with torch.no_grad():
        # 1. Parallel Mode
        out_p = model(x, mode='p', listT=listT, static_feats=static)
        # Output is predicted dist for next steps. 
        # Typically x[t] -> predicts x[t+1]
        
        # 2. Inference Mode (Autoregressive loop internally in model)
        # We need to test if manually feeding steps matches p-mode
        # The model.forward(mode='i') provided implements specific rollout logic
        # Let's test the specific 'i' mode path in the code
        
        # In the provided code, mode='i' uses the whole sequence x as context to generate future
        # To test equivalence, we treat x[:, :1] as context and generate L-1 frames
        
        start_frame = x[:, 0:1]
        future_T = torch.ones(B, L-1, device=device)
        
        # This will run the loop in forward()
        out_i = model(start_frame, mode='i', out_gen_num=L, listT=listT[:, 0:1], listT_future=future_T, static_feats=static)
        
        # Note: out_p contains predictions for t=1..L based on x=0..L-1
        # out_i contains predictions where step 0 is the start frame, and 1..L are generated
        # Comparing them exactly is tricky because 'i' mode feeds its OWN output back, 
        # whereas 'p' mode uses Ground Truth x.
        # To strictly verify, we check shape and graph execution.
        
        shape_match = (out_p.shape == out_i.shape)
        print_status("Inference Shape Match", shape_match, f"{out_p.shape} vs {out_i.shape}")
        
        # Strict Numerical Equivalence Test: Manual Teacher Forcing
        # We manually call internal components step-by-step
        h_ins = None
        outputs = []
        
        emb, _ = model.embedding(x, static_feats=static) # [B, C, L, H, W]
        # Permute for block processing [B, L, C, H, W]
        emb = emb.permute(0, 2, 1, 3, 4)
        
        for t in range(L):
            curr_x = emb[:, t:t+1] # [B, 1, C, H, W]
            curr_T = listT[:, t:t+1]
            
            # Pass through UNet/Blocks
            # Note: This is internal logic simulation
            curr_feat, h_ins = model.convlru_model(curr_x, last_hidden_ins=h_ins, listT=curr_T, cond=model.embedding(x, static)[1])
            
            # Decoder
            res = model.decoder(curr_feat, cond=model.embedding(x, static)[1])
            outputs.append(res)
            
        out_stepwise = torch.cat(outputs, dim=1)
        
        diff = (out_p - out_stepwise).abs().max().item()
        is_consistent = diff < 1e-4
        print_status("Internal State Consistency", is_consistent, f"Max Diff: {diff:.2e}")

def test_flash_fft_fallback():
    print("\n=== 4. FlashFFTConv Fallback Test ===")
    from ModelConvLRU import FlashFFTConvInterface
    device = get_device()
    
    H, W = 64, 64
    fft_layer = FlashFFTConvInterface(16, (H, W)).to(device)
    
    u = torch.randn(2, 16, H, W, device=device)
    k = torch.randn(16, H, W, device=device) # Spatial kernel
    
    try:
        out = fft_layer(u, k)
        print_status("FlashFFT Interface", True, f"Output: {out.shape}")
    except Exception as e:
        print_status("FlashFFT Interface", False, str(e))

if __name__ == "__main__":
    print(f"Running Tests on: {get_device()}")
    
    # 0. Check Triton PScan
    print("\n=== 0. Kernel Check ===")
    if pscan_check():
        print_status("PScan Kernel", True)
    else:
        print_status("PScan Kernel", False)
        sys.exit(1)

    test_configurations()
    test_heads()
    test_consistency()
    test_flash_fft_fallback()
    
    print("\n✅ All Tests Completed.")