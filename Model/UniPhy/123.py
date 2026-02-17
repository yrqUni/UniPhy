"""
精确诊断：逐步增加复杂度找到 inplace 错误的边界。
运行: python diagnose2.py
"""
import torch
import torch.nn as nn
import sys
import os

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}, Device: {device}")

sys.path.insert(0, "/nfs/UniPhy/Model/UniPhy")
sys.path.insert(0, "/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel

# 用较小分辨率加速测试
H, W_img = 49, 60  # 可被 7x15 patch 整除

def test_config(depth, sde_mode, ensemble_size, T, label):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  depth={depth}, sde={sde_mode}, ens={ensemble_size}, T={T}")
    print(f"{'='*60}")
    try:
        model = UniPhyModel(
            in_channels=30, out_channels=30, embed_dim=256, expand=4,
            depth=depth, patch_size=(7, 15),
            img_height=H, img_width=W_img,
            dt_ref=6.0, sde_mode=sde_mode,
            init_noise_scale=0.0001, ensemble_size=ensemble_size,
        ).to(device)

        B = 1
        x = torch.randn(B, T, 30, H, W_img, device=device)
        dt = torch.full((B, T), 6.0, device=device)

        out = model(x, dt)
        loss = out.abs().mean()
        loss.backward()

        print(f"  PASSED ✓")
        del model, x, dt, out, loss
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        err = str(e)
        if "inplace" in err.lower():
            print(f"  FAILED (inplace) ✗")
            print(f"  {err[:200]}")
        else:
            print(f"  FAILED (other) ✗")
            print(f"  {err[:200]}")
        torch.cuda.empty_cache()
        return False

# 逐步增加 depth
for d in [1, 2, 4, 8]:
    if not test_config(d, "ode", 1, 4, f"depth={d}, ODE mode"):
        break

# 如果 depth=8 ODE 通过，测试 SDE
print("\n" + "#"*60)
print("# 切换到 SDE 模式")
print("#"*60)

for d in [1, 2, 4, 8]:
    if not test_config(d, "sde", 1, 4, f"depth={d}, SDE mode"):
        break

# 如果 SDE 通过，测试 ensemble
print("\n" + "#"*60)
print("# 测试 ensemble (模拟 train_step)")
print("#"*60)

for ens in [1, 4]:
    print(f"\n{'='*60}")
    print(f"TEST: ensemble={ens}, depth=8, SDE")
    print(f"{'='*60}")
    try:
        model = UniPhyModel(
            in_channels=30, out_channels=30, embed_dim=256, expand=4,
            depth=8, patch_size=(7, 15),
            img_height=H, img_width=W_img,
            dt_ref=6.0, sde_mode="sde",
            init_noise_scale=0.0001, ensemble_size=ens,
        ).to(device)

        B = 1
        T = 4
        x = torch.randn(B, T+1, 30, H, W_img, device=device)
        dt = torch.full((B, T+1), 6.0, device=device)

        x_input = x[:, :-1]
        x_target = x[:, 1:]
        dt_input = dt[:, 1:]

        member_idx = torch.randint(0, max(1,ens), (B,), device=device) if ens > 1 else None
        out = model(x_input, dt_input, member_idx=member_idx)
        out_real = out.real if out.is_complex() else out
        loss = (out_real - x_target).abs().mean()

        if ens > 1:
            # 模拟 train_step 中的 ensemble 采样
            with torch.no_grad():
                for _ in range(ens - 1):
                    ridx = torch.randint(0, ens, (B,), device=device)
                    out2 = model(x_input, dt_input, member_idx=ridx)

        loss.backward()
        print(f"  PASSED ✓")
        del model
        torch.cuda.empty_cache()
    except RuntimeError as e:
        err = str(e)
        if "inplace" in err.lower():
            print(f"  FAILED (inplace) ✗")
            print(f"  {err[:300]}")
        else:
            print(f"  FAILED (other) ✗")
            print(f"  {err[:300]}")
        torch.cuda.empty_cache()

# 测试 forward_rollout (align)
print("\n" + "#"*60)
print("# 测试 forward_rollout (align pipeline)")
print("#"*60)

try:
    model = UniPhyModel(
        in_channels=30, out_channels=30, embed_dim=256, expand=4,
        depth=8, patch_size=(7, 15),
        img_height=H, img_width=W_img,
        dt_ref=6.0, sde_mode="sde",
        init_noise_scale=0.0001, ensemble_size=1,
    ).to(device)

    B = 1
    x_ctx = torch.randn(B, 4, 30, H, W_img, device=device)
    dt_ctx = torch.full((B, 4), 6.0, device=device)
    dt_list = [torch.full((B,), 3.0, device=device) for _ in range(4)]

    pred = model.forward_rollout(x_ctx, dt_ctx, dt_list)
    loss = pred.abs().mean()
    loss.backward()
    print(f"  forward_rollout: PASSED ✓")
except RuntimeError as e:
    err = str(e)
    print(f"  forward_rollout: FAILED ✗")
    print(f"  {err[:300]}")

print("\n" + "="*60)
print("诊断完成")
print("="*60)