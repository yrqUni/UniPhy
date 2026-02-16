"""
独立诊断脚本：测试 ComplexSVDTransform + Block 的 backward 是否出错。
在你的环境中运行: python diagnose_inplace.py
它会逐步隔离出问题出在哪个模块。
"""
import torch
import torch.nn as nn
import sys

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}, Device: {device}")

# ============================================================
# TEST 1: 纯 ComplexSVDTransform
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: ComplexSVDTransform encode → decode → backward")
print("=" * 60)

dim = 256
dft = torch.exp(-2j * torch.pi
                * torch.arange(dim).float().unsqueeze(1)
                * torch.arange(dim).float().unsqueeze(0)
                / dim) / (dim ** 0.5)
dft_inv = dft.conj().T.contiguous()

w_re = nn.Parameter(dft.real.clone().to(device))
w_im = nn.Parameter(dft.imag.clone().to(device))
w_inv_re = nn.Parameter(dft_inv.real.clone().to(device))
w_inv_im = nn.Parameter(dft_inv.imag.clone().to(device))

x = torch.randn(2, 4, dim, device=device)

# 无 clone
try:
    W = torch.complex(w_re, w_im)
    W_inv = torch.complex(w_inv_re, w_inv_im)
    x_c = torch.complex(x, torch.zeros_like(x))
    encoded = torch.einsum("...d,de->...e", x_c, W)
    decoded = torch.einsum("...d,de->...e", encoded, W_inv)
    loss = decoded.abs().sum()
    loss.backward()
    print("  无 clone: PASSED")
except RuntimeError as e:
    print(f"  无 clone: FAILED - {e}")

# 清除梯度
w_re.grad = None; w_im.grad = None
w_inv_re.grad = None; w_inv_im.grad = None

# 有 clone
try:
    W = torch.complex(w_re, w_im).clone()
    W_inv = torch.complex(w_inv_re, w_inv_im).clone()
    x_c = torch.complex(x, torch.zeros_like(x))
    encoded = torch.einsum("...d,de->...e", x_c, W)
    decoded = torch.einsum("...d,de->...e", encoded, W_inv)
    loss = decoded.abs().sum()
    loss.backward()
    print("  有 clone: PASSED")
except RuntimeError as e:
    print(f"  有 clone: FAILED - {e}")

# ============================================================
# TEST 2: 模拟完整 Block.forward 中的所有操作
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: 模拟 Block forward 中 W/W_inv 的实际使用模式")
print("=" * 60)

w_re.grad = None; w_im.grad = None
w_inv_re.grad = None; w_inv_im.grad = None

try:
    W = torch.complex(w_re, w_im).clone()
    W_inv = torch.complex(w_inv_re, w_inv_im).clone()

    # encode: x → x_eigen
    x_c = torch.complex(x, torch.zeros_like(x))
    x_eigen = torch.einsum("...d,de->...e", x_c, W)

    # 中间有大量运算（flux tracking, pscan 等），结果作为 u_out
    u_out = x_eigen * 0.9 + torch.randn_like(x_eigen) * 0.1

    # decode: u_out → decoded
    decoded = torch.einsum("...d,de->...e", u_out, W_inv)

    loss = decoded.abs().sum()
    loss.backward()
    print("  PASSED")
except RuntimeError as e:
    print(f"  FAILED - {e}")

# ============================================================
# TEST 3: 导入实际模块测试
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: 导入实际 UniPhyOps 模块")
print("=" * 60)

try:
    sys.path.insert(0, "/nfs/UniPhy/Model/UniPhy")
    from UniPhyOps import ComplexSVDTransform
    basis = ComplexSVDTransform(dim).to(device)
    W, W_inv = basis.get_matrix(torch.complex64)
    print(f"  W shape: {W.shape}, W._version: {W._version}")
    print(f"  W_inv shape: {W_inv.shape}, W_inv._version: {W_inv._version}")
    print(f"  W requires_grad: {W.requires_grad}")

    x_c = torch.complex(x, torch.zeros_like(x))
    encoded = basis.encode_with(x_c, W)
    decoded = basis.decode_with(encoded, W_inv)
    loss = decoded.abs().sum()
    loss.backward()
    print("  ComplexSVDTransform standalone: PASSED")
except RuntimeError as e:
    print(f"  FAILED - {e}")
except ImportError as e:
    print(f"  Cannot import: {e}")

# ============================================================
# TEST 4: 完整模型单步测试
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: 完整 UniPhyModel 单步前向 + backward")
print("=" * 60)

try:
    sys.path.insert(0, "/nfs/UniPhy/Exp/ERA5")
    from ModelUniPhy import UniPhyModel
    model = UniPhyModel(
        in_channels=30, out_channels=30, embed_dim=256, expand=4,
        depth=2,  # 先用2层测试
        patch_size=(7, 15), img_height=721, img_width=1440,
        dt_ref=6.0, sde_mode="sde", init_noise_scale=0.0001,
        ensemble_size=1,
    ).to(device)

    x_in = torch.randn(1, 2, 30, 721, 1440, device=device)
    dt_in = torch.full((1, 2), 6.0, device=device)
    out = model(x_in, dt_in)
    loss = out.abs().mean()
    loss.backward()
    print("  depth=2: PASSED")
except RuntimeError as e:
    print(f"  FAILED - {e}")
except Exception as e:
    print(f"  ERROR - {type(e).__name__}: {e}")

# ============================================================
# TEST 5: 检查版本号追踪
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: 版本号追踪")
print("=" * 60)

w_re2 = nn.Parameter(torch.randn(256, 256, device=device))
w_im2 = nn.Parameter(torch.randn(256, 256, device=device))
print(f"  初始: w_re._version={w_re2._version}, w_im._version={w_im2._version}")

c1 = torch.complex(w_re2, w_im2)
print(f"  torch.complex后: w_re._version={w_re2._version}, c1._version={c1._version}")

c2 = c1.clone()
print(f"  clone后: c2._version={c2._version}, c1._version={c1._version}")

r = torch.einsum("ij,jk->ik", c2, c2)
print(f"  einsum后: w_re._version={w_re2._version}")

loss = r.abs().sum()
loss.backward()
print(f"  backward后: w_re._version={w_re2._version}")

print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)
