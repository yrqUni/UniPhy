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
        "input_size": [(32, 32), (32, 48)],
        "convlru_num_blocks": [1, 2],
        "emb_ch": [16, 32],
    }

    fixed_args = {
        "input_ch": 1,
        "out_ch": 1,
        "lru_rank": 8,
        "ffn_ratio": 2.0,
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total = len(combinations)
    passed = 0
    failed = 0

    print(f"Total combinations to check: {total}")
    print("-" * 60)

    for i, params in enumerate(combinations):
        args_dict = {**fixed_args, **params}
        args = SimpleNamespace(**args_dict)

        config_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i+1}/{total}] Checking: {config_str}")

        try:
            model = UniPhy(args).to(device)

            B, L, C = 2, 4, args.input_ch
            H, W = args.input_size

            x = torch.randn(B, L, C, H, W, device=device)
            listT = torch.ones(B, L, device=device) * 0.1

            if args.dist_mode == "diffusion":
                noise = torch.randn(B, L, args.out_ch, H, W, device=device)
                t = torch.randint(0, 100, (B * L,), device=device).long()
                out_p, _ = model(x, mode='p', listT=listT, x_noisy=noise, t=t)
                expected_out_ch = args.out_ch
            elif args.dist_mode in ["gaussian", "laplace"]:
                out_p, _ = model(x, mode='p', listT=listT)
                expected_out_ch = args.out_ch * 2
            else:
                out_p, _ = model(x, mode='p', listT=listT)
                expected_out_ch = args.out_ch

            expected_shape = (B, expected_out_ch, L, H, W)
            assert out_p.shape == expected_shape, \
                f"Train Output shape mismatch: got {out_p.shape}, expected {expected_shape}"

            loss = out_p.sum()
            loss.backward()

            out_gen_num = 3
            future_T = torch.ones(B, out_gen_num - 1, device=device) * 0.1

            out_i, _ = model(x, mode='i', out_gen_num=out_gen_num, listT=listT, listT_future=future_T)

            expected_infer_shape = (B, expected_out_ch, out_gen_num, H, W)
            assert out_i.shape == expected_infer_shape, \
                f"Infer Output shape mismatch: got {out_i.shape}, expected {expected_infer_shape}"

            print(f"    -> PASS")
            passed += 1

        except Exception:
            print(f"    -> FAIL")
            traceback.print_exc()
            failed += 1
            print("\n!!! Stopping early to allow debugging of the first error !!!")
            sys.exit(1)

        finally:
            if 'model' in locals(): del model
            if 'x' in locals(): del x
            if 'listT' in locals(): del listT
            if 'loss' in locals(): del loss
            if 'out_p' in locals(): del out_p
            if 'out_i' in locals(): del out_i
            torch.cuda.empty_cache()

    print("-" * 60)
    print(f"Summary: {passed} Passed, {failed} Failed")

if __name__ == "__main__":
    main()

