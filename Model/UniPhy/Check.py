import torch
import sys
import traceback
from ModelUniPhy import UniPhy

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_configuration(config_name, args, device):
    print(f"Testing configuration: {config_name}")
    try:
        model = UniPhy(args).to(device)
        # Input: (B, L, C, H, W)
        B, L, C, H, W = 2, 3, args.input_ch, args.input_size[0], args.input_size[1]
        x = torch.randn(B, L, C, H, W).to(device)
        listT = torch.ones(B, L).to(device)

        if args.dist_mode == "diffusion":
            out_ch = args.out_ch
            # Fix: x_noisy resolution should match input resolution (H, W), not scaled up
            x_noisy = torch.randn(B, L, out_ch, H, W).to(device)
            t = torch.randint(0, 1000, (B * L,)).float().to(device)
            out, _ = model(x, mode="p", listT=listT, x_noisy=x_noisy, t=t)
            # Fix: Output resolution matches input resolution
            expected_shape = (B, L, out_ch, H, W)
        elif args.dist_mode == "gaussian":
            out, _ = model(x, mode="p", listT=listT)
            # Fix: Output resolution matches input resolution
            expected_shape = (B, L, args.out_ch * 2, H, W)
        else:
            out, _ = model(x, mode="p", listT=listT)
            expected_shape = (B, L, args.out_ch, H, W)

        assert out.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out.shape}"
        
        # Test inference mode
        out_gen, _ = model(x, mode="inference", out_gen_num=2, listT=listT)
        print(f"  [Pass] {config_name}")
        return True
    except Exception as e:
        print(f"  [Fail] {config_name}")
        traceback.print_exc()
        return False

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests as Triton requires CUDA.")
        return

    device = torch.device("cuda")
    torch.set_default_device(device)

    base_config = {
        "input_ch": 2,
        "input_size": (32, 64),
        "emb_ch": 16,
        "hidden_factor": (2, 2),
        "convlru_num_blocks": 2,
        "lru_rank": 8,
        "dt_ref": 1.0,
        "ffn_ratio": 2.0,
        "ConvType": "conv",
        "spectral_modes_h": 8,
        "spectral_modes_w": 8,
        "out_ch": 2,
    }

    arch_modes = ["unet", "no_unet"]
    down_modes = ["avg", "conv", "shuffle"]
    dist_modes = ["gaussian", "diffusion"]
    
    configs = []
    
    for arch in arch_modes:
        for dist in dist_modes:
            if arch == "unet":
                for down in down_modes:
                    name = f"Arch={arch}, Down={down}, Dist={dist}"
                    cfg = base_config.copy()
                    cfg["Arch"] = arch
                    cfg["down_mode"] = down
                    cfg["dist_mode"] = dist
                    configs.append((name, MockArgs(**cfg)))
            else:
                name = f"Arch={arch}, Dist={dist}"
                cfg = base_config.copy()
                cfg["Arch"] = arch
                cfg["down_mode"] = "avg" 
                cfg["dist_mode"] = dist
                configs.append((name, MockArgs(**cfg)))

    total = len(configs)
    passed = 0
    print(f"Starting check for {total} configurations...")
    print("-" * 60)

    for name, args in configs:
        if test_configuration(name, args, device):
            passed += 1

    print("-" * 60)
    print(f"Test Summary: {passed}/{total} Passed")
    
    if passed == total:
        print("All checks passed successfully.")
        sys.exit(0)
    else:
        print("Some checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

