import torch
import itertools
import sys
import traceback
from types import SimpleNamespace
from ModelUniPhy import UniPhy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running checks on {device}")

    param_grid = {
        "Arch": ["unet", "no_unet"],
        "down_mode": ["avg", "conv", "shuffle"],
        "dist_mode": ["gaussian", "laplace", "diffusion"],
        "ConvType": ["conv", "dcn"],
        "hidden_factor": [(1, 1), (2, 2)],
        "convlru_num_blocks": [1, 2],
    }

    fixed_args = {
        "input_ch": 1,
        "input_size": (32, 32),
        "emb_ch": 16,
        "lru_rank": 8,
        "ffn_ratio": 2.0,
        "out_ch": 1,
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total = len(combinations)
    passed = 0
    failed = 0

    print(f"Total combinations to check: {total}")

    for i, params in enumerate(combinations):
        args_dict = {**fixed_args, **params}
        args = SimpleNamespace(**args_dict)
        
        config_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i+1}/{total}] Checking: {config_str}")

        try:
            model = UniPhy(args).to(device)
            
            B, C, L, H, W = 2, args.input_ch, 4, args.input_size[0], args.input_size[1]
            x = torch.randn(B, C, L, H, W, device=device)
            listT = torch.ones(B, L, device=device) * 0.1

            if args.dist_mode in ["gaussian", "laplace"]:
                expected_ch = args.out_ch * 2
            else:
                expected_ch = args.out_ch

            out_p, _ = model(x, mode='p', listT=listT)
            
            assert out_p.shape == (B, expected_ch, L, H, W), \
                f"Output shape mismatch (training): got {out_p.shape}, expected {(B, expected_ch, L, H, W)}"
            
            loss = out_p.sum()
            loss.backward()

            out_gen_num = 3
            future_T = torch.ones(B, out_gen_num - 1, device=device) * 0.1
            out_i, _ = model(x, mode='i', out_gen_num=out_gen_num, listT=listT, listT_future=future_T)
            
            assert out_i.shape == (B, expected_ch, out_gen_num, H, W), \
                f"Inference shape mismatch: got {out_i.shape}, expected {(B, expected_ch, out_gen_num, H, W)}"

            print(f"    -> PASS")
            passed += 1
            
            del model, x, listT, out_p, loss, out_i
            torch.cuda.empty_cache()

        except Exception:
            print(f"    -> FAIL")
            traceback.print_exc()
            failed += 1
            if failed >= 1: 
                print("\n!!! Stopping early to allow debugging of the first error !!!")
                sys.exit(1)

    print("-" * 50)
    print(f"Summary: {passed} Passed, {failed} Failed")
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

