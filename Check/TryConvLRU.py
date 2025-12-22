import os
import sys
import random
from typing import Optional, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU

try:
    from pscan import pscan_check
except Exception:
    pscan_check = None


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_tf32(enable: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = bool(enable)
    torch.backends.cudnn.allow_tf32 = bool(enable)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enable else "highest")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_status(name: str, passed: bool, msg: str = ""):
    if passed:
        print(f"\033[92m[PASS] {name:<34}\033[0m {msg}")
    else:
        print(f"\033[91m[FAIL] {name:<34}\033[0m {msg}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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
        self.convlru_num_blocks = 2
        self.ffn_hidden_ch = 32
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
        self.use_checkpointing = True


def expected_out_channels(args: MockArgs) -> int:
    if str(args.head_mode).lower() == "gaussian":
        return int(args.out_ch) * 2
    return int(args.out_ch)


def make_inputs(
    device: torch.device,
    B: int,
    L: int,
    C: int,
    H: int,
    W: int,
    static_ch: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x = torch.randn(B, L, C, H, W, device=device, dtype=dtype)
    static = None
    if static_ch > 0:
        static = torch.randn(B, static_ch, H, W, device=device, dtype=dtype)
    return x, static


def test_pscan_kernel():
    print("\n=== 0. Kernel Check ===")
    if pscan_check is None:
        print_status("PScan Kernel", True, "pscan_check not available; skipped")
        return
    try:
        ok = bool(pscan_check(batch_size=2, seq_length=16, channels=4, state_dim=8))
        print_status("PScan Kernel", ok, "")
    except Exception as e:
        print_status("PScan Kernel", False, str(e))


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
    shapes = [
        (2, 4, 4, 32, 32),
        (1, 1, 4, 32, 32),
        (2, 7, 4, 32, 32),
    ]
    dtypes = [torch.float32]
    if device.type == "cuda":
        dtypes.append(torch.float16)

    for B, L, C, H, W in shapes:
        for dtype in dtypes:
            x, static = make_inputs(device, B, L, C, H, W, static_ch=2, dtype=dtype)
            listT = torch.ones(B, L, device=device, dtype=dtype)

            for name, unet, bidir, sel, n_exp in configs:
                tag = f"{name}|B{B}L{L}|{str(dtype).replace('torch.', '')}"
                try:
                    args = MockArgs()
                    args.unet = bool(unet)
                    args.bidirectional = bool(bidir)
                    args.use_selective = bool(sel)
                    args.num_expert = int(n_exp)
                    model = ConvLRU(args).to(device)
                    model.train()

                    use_amp = device.type == "cuda" and dtype == torch.float16
                    
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                        out = model(x, mode="p", listT=listT, static_feats=static)
                        expC = expected_out_channels(args)
                        exp_shape = (B, L, expC, H, W)
                        if not isinstance(out, torch.Tensor) or tuple(out.shape) != exp_shape:
                            raise ValueError(
                                f"Shape mismatch: got {tuple(out.shape) if isinstance(out, torch.Tensor) else type(out)} expected {exp_shape}"
                            )
                        loss = out.float().pow(2).mean()

                    loss.backward()
                    print_status(tag, True, f"Params: {count_params(model)}")
                except Exception as e:
                    print_status(tag, False, str(e))


def test_heads():
    print("\n=== 2. Decoder Heads Test ===")
    device = get_device()
    B, L, C, H, W = 2, 4, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    modes = ["gaussian", "diffusion", "token"]

    for mode in modes:
        try:
            args = MockArgs()
            args.head_mode = mode
            model = ConvLRU(args).to(device)
            model.train()
            timestep = None
            if mode == "diffusion":
                timestep = torch.randint(0, 1000, (B,), device=device).float()
            out = model(x, mode="p", listT=None, static_feats=None, timestep=timestep)
            if mode == "token":
                if not (isinstance(out, tuple) and len(out) == 3):
                    raise ValueError(f"Token head must return tuple(len=3), got {type(out)}")
                quant, vq_loss, idx = out
                ok = tuple(quant.shape) == (B, L, args.out_ch, H, W) and vq_loss.dim() == 0 and idx.dim() == 2
                (quant.sum() + vq_loss).backward()
                print_status(f"Head: {mode}", ok, f"Quant: {tuple(quant.shape)} Loss: {vq_loss.item():.4f}")
            else:
                exp_shape = (B, L, expected_out_channels(args), H, W)
                ok = isinstance(out, torch.Tensor) and tuple(out.shape) == exp_shape
                out.sum().backward()
                print_status(f"Head: {mode}", ok, f"Shape: {tuple(out.shape)}")
        except Exception as e:
            print_status(f"Head: {mode}", False, str(e))


def test_diffusion_head_odd_dim_error():
    print("\n=== 2.1 DiffusionHead Odd-Dim Safety Test ===")
    device = get_device()
    try:
        args = MockArgs()
        args.head_mode = "diffusion"
        args.dec_hidden_layers_num = 1
        args.dec_hidden_ch = 15
        _ = ConvLRU(args).to(device)
        print_status("Odd dim should fail", False, "Expected ValueError but model constructed")
    except ValueError as e:
        print_status("Odd dim should fail", True, str(e))
    except Exception as e:
        print_status("Odd dim should fail", False, str(e))


def test_listT_none_all_modes():
    print("\n=== 3. listT=None Paths Test ===")
    device = get_device()
    B, L, C, H, W = 1, 6, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    static = torch.randn(B, 2, H, W, device=device, dtype=torch.float32)

    configs = [("Flat", False), ("UNet", True)]
    for name, unet in configs:
        try:
            args = MockArgs()
            args.unet = bool(unet)
            args.num_expert = 4
            model = ConvLRU(args).to(device)
            model.eval()
            with torch.no_grad():
                out = model(x, mode="p", listT=None, static_feats=static)
            exp_shape = (B, L, expected_out_channels(args), H, W)
            ok = isinstance(out, torch.Tensor) and tuple(out.shape) == exp_shape
            print_status(f"{name}: p listT=None", ok, f"Shape: {tuple(out.shape)}")
        except Exception as e:
            print_status(f"{name}: p listT=None", False, str(e))

        try:
            args = MockArgs()
            args.unet = bool(unet)
            args.num_expert = 4
            model = ConvLRU(args).to(device)
            model.eval()
            with torch.no_grad():
                out_i = model(x[:, 0:1], mode="i", out_gen_num=L, listT=None, listT_future=None, static_feats=static)
            exp_shape_i = (B, L, expected_out_channels(args), H, W)
            ok = isinstance(out_i, torch.Tensor) and tuple(out_i.shape) == exp_shape_i
            print_status(f"{name}: i listT=None", ok, f"Shape: {tuple(out_i.shape)}")
        except Exception as e:
            print_status(f"{name}: i listT=None", False, str(e))


def test_consistency_step1():
    print("\n=== 4. Consistency (Parallel Step-1 vs Inference Step-1) ===")
    device = get_device()
    args = MockArgs()
    args.unet = True
    args.bidirectional = False
    args.num_expert = 4
    model = ConvLRU(args).to(device)
    model.eval()

    B, L, C, H, W = 1, 6, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    static = torch.randn(B, 2, H, W, device=device, dtype=torch.float32)
    listT = torch.ones(B, L, device=device, dtype=torch.float32)
    start_frame = x[:, 0:1]
    future_T = torch.ones(B, L - 1, device=device, dtype=torch.float32)

    with torch.no_grad():
        out_p_step1 = model(start_frame, mode="p", listT=listT[:, 0:1], static_feats=static)
        out_i = model(
            start_frame,
            mode="i",
            out_gen_num=L,
            listT=listT[:, 0:1],
            listT_future=future_T,
            static_feats=static,
        )

    shape_ok = tuple(out_p_step1.shape) == tuple(out_i[:, 0:1].shape)
    diff = (out_p_step1[:, 0] - out_i[:, 0]).abs().max().item() if shape_ok else float("inf")
    print_status("Shape Match", shape_ok, f"{tuple(out_p_step1.shape)} vs {tuple(out_i[:, 0:1].shape)}")
    print_status("Step-1 MaxDiff<1e-4", diff < 1e-4, f"MaxDiff: {diff:.2e}")


def test_backward_sanity():
    print("\n=== 5. Backward Sanity (Multiple Steps) ===")
    device = get_device()
    args = MockArgs()
    args.unet = True
    args.num_expert = 4
    model = ConvLRU(args).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B, L, C, H, W = 1, 6, 4, 32, 32
    x = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    static = torch.randn(B, 2, H, W, device=device, dtype=torch.float32)
    listT = torch.ones(B, L, device=device, dtype=torch.float32)

    try:
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(x, mode="p", listT=listT, static_feats=static)
            loss = out.float().pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print_status("Backward/Step", True, "3 steps ok")
    except Exception as e:
        print_status("Backward/Step", False, str(e))


def test_large_hw_smoke():
    print("\n=== 6. Large H,W Smoke ===")
    device = get_device()
    args = MockArgs()
    args.input_size = (721, 1440)
    args.input_ch = 30
    args.out_ch = 30
    args.static_ch = 6
    args.emb_ch = 32
    args.emb_hidden_ch = 32
    args.emb_hidden_layers_num = 1
    args.hidden_factor = (7, 12)
    args.convlru_num_blocks = 2
    args.ffn_hidden_ch = 32
    args.ffn_hidden_layers_num = 1
    args.use_selective = True
    args.bidirectional = True
    args.use_freq_prior = True
    args.use_sh_prior = False
    args.unet = True
    args.num_expert = 4
    args.activate_expert = 2
    args.head_mode = "gaussian"
    args.dec_strategy = "pxsf"
    args.dec_hidden_layers_num = 0

    B, L = 1, 2
    C = args.input_ch
    H, W = args.input_size
    x = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    static = torch.randn(B, args.static_ch, H, W, device=device, dtype=torch.float32)
    listT = torch.ones(B, L, device=device, dtype=torch.float32)

    try:
        model = ConvLRU(args).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x, mode="p", listT=listT, static_feats=static)
        ok = isinstance(out, torch.Tensor) and tuple(out.shape) == (B, L, args.out_ch * 2, H, W)
        print_status("Large p forward", ok, f"Shape: {tuple(out.shape)}")
    except Exception as e:
        print_status("Large p forward", False, str(e))


if __name__ == "__main__":
    seed_all(42)
    enable_tf32(True)
    device = get_device()
    print(f"Running Tests on: {device}")

    test_pscan_kernel()
    test_configurations()
    test_heads()
    test_diffusion_head_odd_dim_error()
    test_listT_none_all_modes()
    test_consistency_step1()
    test_backward_sanity()
    test_large_hw_smoke()

    print("\n✅ All Tests Completed.")
