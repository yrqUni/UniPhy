import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU"))
from ModelConvLRU import ConvLRU
from pscan import pscan_check

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
        self.bidirectional = False
        self.use_selective = False
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.num_expert = -1
        self.activate_expert = 2
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        self.head_mode = "gaussian"
        self.unet = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_status(name, passed, msg=""):
    if passed:
        print(f"\033[92m[PASS] {name:<25}\033[0m {msg}")
    else:
        print(f"\033[91m[FAIL] {name:<25}\033[0m {msg}")


def test_configurations():
    print("\n=== 1. Architecture Combinations Test ===")
    device = get_device()
    configs = [
        ("Base_Flat", False, False, False, -1),
        ("UNet_Only", True, False, False, -1),
        ("BiDir_Only", False, True, False, -1),
        ("Selective_Only", False, False, True, -1),
        ("Full_Advanced", True, True, True, -1),
        ("MoE_Base", False, False, False, 4),
        ("MoE_UNet", True, False, False, 4),
    ]
    B, L, C, H, W = 2, 8, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    for name, unet, bidir, sel, n_exp in configs:
        try:
            args = MockArgs()
            args.unet = unet
            args.bidirectional = bidir
            args.use_selective = sel
            args.num_expert = n_exp
            model = ConvLRU(args).to(device)
            out = model(x, mode="p", listT=listT, static_feats=static)
            expected_C = args.out_ch * 2
            expected_shape = (B, L, expected_C, H, W)
            if out.shape != expected_shape:
                raise ValueError(f"Shape Mismatch: Got {out.shape}, Expected {expected_shape}")
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
            out = model(x, mode="p", listT=None, static_feats=None, timestep=timestep)
            passed = False
            msg = ""
            if mode == "gaussian":
                if out.shape[2] == args.out_ch * 2:
                    passed = True
                    msg = f"Shape: {out.shape}"
            elif mode == "diffusion":
                if out.shape[2] == args.out_ch:
                    passed = True
                    msg = f"Shape: {out.shape}"
            elif mode == "token":
                if isinstance(out, tuple) and len(out) == 3:
                    quant, vq_loss, idx = out
                    if quant.shape[2] == args.out_ch:
                        loss = quant.sum() + vq_loss
                        loss.backward()
                        passed = True
                        msg = f"VQ Shape: {quant.shape}, Loss: {vq_loss.item():.4f}"
            if not passed:
                msg = f"Output Check Failed. Shape: {out.shape if not isinstance(out, tuple) else out[0].shape}"
            print_status(f"Head: {mode}", passed, msg)
        except Exception as e:
            print_status(f"Head: {mode}", False, str(e))


def test_consistency():
    print("\n=== 3. Consistency (Parallel vs Inference) ===")
    device = get_device()
    args = MockArgs()
    args.unet = True
    args.bidirectional = False
    args.num_expert = 4
    model = ConvLRU(args).to(device)
    model.eval()
    B, L, C, H, W = 1, 6, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    start_frame = x[:, 0:1]
    with torch.no_grad():
        out_p = model(start_frame, mode="p", listT=listT[:, 0:1], static_feats=static)
        future_T = torch.ones(B, L - 1, device=device)
        out_i = model(
            start_frame,
            mode="i",
            out_gen_num=L,
            listT=listT[:, 0:1],
            listT_future=future_T,
            static_feats=static,
        )
        shape_match = (out_p.shape[0], 1, out_p.shape[2], out_p.shape[3], out_p.shape[4]) == (
            out_i.shape[0],
            1,
            out_i.shape[2],
            out_i.shape[3],
            out_i.shape[4],
        )
        print_status("Inference Shape Match", shape_match, f"{out_p.shape} vs {out_i.shape}")
        diff_step1 = (out_p[:, 0] - out_i[:, 0]).abs().max().item()
        is_consistent = diff_step1 < 1e-4
        print_status("Step-1 Consistency", is_consistent, f"Max Diff: {diff_step1:.2e}")


def test_flash_fft_fallback():
    print("\n=== 4. FlashFFTConv Fallback Test ===")
    from ModelConvLRU import FlashFFTConvInterface

    device = get_device()
    H, W = 64, 64
    fft_layer = FlashFFTConvInterface(16, (H, W)).to(device)
    u = torch.randn(2, 16, H, W, device=device)
    k = torch.randn(16, H, W, device=device)
    try:
        out = fft_layer(u, k)
        print_status("FlashFFT Interface", True, f"Output: {out.shape}")
    except Exception as e:
        print_status("FlashFFT Interface", False, str(e))


if __name__ == "__main__":
    print(f"Running Tests on: {get_device()}")
    print("\n=== 0. Kernel Check ===")
    try:
        if pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8):
            print_status("PScan Kernel", True)
        else:
            print_status("PScan Kernel", False)
            sys.exit(1)
    except Exception as e:
        print_status("PScan Kernel", False, str(e))
        sys.exit(1)
    test_configurations()
    test_heads()
    test_consistency()
    test_flash_fft_fallback()
    print("\n✅ All Tests Completed.")
