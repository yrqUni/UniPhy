import os
import sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU


def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
        self.use_gate = True

        self.head_mode = "gaussian"
        self.out_ch = 4
        self.dec_hidden_ch = 16
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_status(name, passed, msg=""):
    if passed:
        print(f"\033[92m[PASS] {name:<32}\033[0m {msg}")
    else:
        print(f"\033[91m[FAIL] {name:<32}\033[0m {msg}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def expected_out_channels(args: MockArgs):
    if args.head_mode == "gaussian":
        return args.out_ch * 2
    return args.out_ch


def make_inputs(device, B=2, L=4, C=4, H=32, W=32, static_ch=2):
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, static_ch, H, W, device=device) if static_ch > 0 else None
    listT = torch.ones(B, L, device=device)
    return x, static, listT


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
    B, L, C, H, W = 2, 4, 4, 32, 32
    x, static, listT = make_inputs(device, B=B, L=L, C=C, H=H, W=W, static_ch=2)
    for name, unet, bidir, sel, n_exp in configs:
        try:
            args = MockArgs()
            args.unet = unet
            args.bidirectional = bidir
            args.use_selective = sel
            args.num_expert = n_exp
            model = ConvLRU(args).to(device)
            model.train()
            out = model(x, mode="p", listT=listT, static_feats=static)
            expected_shape = (B, L, expected_out_channels(args), H, W)
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"Shape mismatch: got {tuple(out.shape)} expected {expected_shape}")
            out.sum().backward()
            print_status(name, True, f"Params: {count_params(model)}")
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
            model.train()
            timestep = None
            if mode == "diffusion":
                timestep = torch.randint(0, 1000, (B,), device=device).float()
            out = model(x, mode="p", listT=None, static_feats=None, timestep=timestep)
            if mode == "gaussian":
                passed = tuple(out.shape) == (B, L, args.out_ch * 2, H, W)
                msg = f"Shape: {tuple(out.shape)}"
                out.sum().backward()
            elif mode == "diffusion":
                passed = tuple(out.shape) == (B, L, args.out_ch, H, W)
                msg = f"Shape: {tuple(out.shape)}"
                out.sum().backward()
            else:
                if isinstance(out, tuple) and len(out) == 3:
                    quant, vq_loss, idx = out
                    passed = tuple(quant.shape) == (B, L, args.out_ch, H, W) and vq_loss.dim() == 0
                    msg = f"VQ Shape: {tuple(quant.shape)}, Loss: {float(vq_loss):.4f}, Idx: {tuple(idx.shape)}"
                    (quant.sum() + vq_loss).backward()
                else:
                    passed = False
                    msg = f"Unexpected output type: {type(out)}"
            print_status(f"Head: {mode}", passed, msg)
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
    future_T = torch.ones(B, L - 1, device=device)
    with torch.no_grad():
        out_p_full = model(x, mode="p", listT=listT, static_feats=static)
        out_p_step1 = model(start_frame, mode="p", listT=listT[:, 0:1], static_feats=static)
        out_i = model(
            start_frame,
            mode="i",
            out_gen_num=L,
            listT=listT[:, 0:1],
            listT_future=future_T,
            static_feats=static,
        )
        print_status("Parallel Full Shape", True, f"{tuple(out_p_full.shape)}")
        shape_match = out_p_step1.shape == out_i[:, 0:1].shape
        print_status("Inference Shape Match", shape_match, f"{tuple(out_p_step1.shape)} vs {tuple(out_i[:, 0:1].shape)}")
        diff_step1 = (out_p_step1[:, 0] - out_i[:, 0]).abs().max().item()
        is_consistent = diff_step1 < 1e-4
        print_status("Step-1 Consistency", is_consistent, f"Max Diff: {diff_step1:.2e}")


def test_inference_listT_none():
    print("\n=== 3.1 Inference listT=None Path Test ===")
    device = get_device()
    args = MockArgs()
    args.unet = True
    args.num_expert = 4
    model = ConvLRU(args).to(device)
    model.eval()
    B, out_gen_num, C, H, W = 1, 6, 4, 32, 32
    start_frame = torch.randn(B, 1, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    try:
        with torch.no_grad():
            out_i = model(
                start_frame,
                mode="i",
                out_gen_num=out_gen_num,
                listT=None,
                listT_future=None,
                static_feats=static,
            )
        expected_C = expected_out_channels(args)
        ok = tuple(out_i.shape) == (B, out_gen_num, expected_C, H, W)
        print_status("Inference listT=None", ok, f"Shape: {tuple(out_i.shape)}")
    except Exception as e:
        print_status("Inference listT=None", False, str(e))


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
    x = torch.randn(B, L, C, H, W, device=device)
    static = torch.randn(B, 2, H, W, device=device)
    listT = torch.ones(B, L, device=device)
    try:
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            out = model(x, mode="p", listT=listT, static_feats=static)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
        print_status("Backward/Step", True, "3 steps ok")
    except Exception as e:
        print_status("Backward/Step", False, str(e))


if __name__ == "__main__":
    seed_all(42)
    enable_tf32()
    device = get_device()
    print(f"Running Tests on: {device}")

    test_configurations()
    test_heads()
    test_diffusion_head_odd_dim_error()
    test_consistency()
    test_inference_listT_none()
    test_backward_sanity()

    print("\n✅ All Tests Completed.")
