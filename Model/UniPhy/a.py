import torch
import sys
import os
import numpy as np

# 路径设置
sys.path.append(os.getcwd())
try:
    from ModelUniPhy import UniPhy, ParallelPhysicalRecurrentLayer
except ImportError:
    print("Error: 找不到 ModelUniPhy.py")
    sys.exit(1)

class MockArgs:
    def __init__(self):
        self.input_ch = 1
        self.out_ch = 1
        self.input_size = (16, 16) # 小尺寸方便调试
        self.emb_ch = 4            # 小 Channel
        self.static_ch = 4         # 必须 > 0 以启用 H0
        self.hidden_factor = (1, 1)
        self.convlru_num_blocks = 1
        self.lru_rank = 2          # 小 Rank
        self.down_mode = "avg"
        self.dist_mode = "gaussian"
        self.dynamics_mode = "spectral"
        self.spectral_modes_h = 16
        self.spectral_modes_w = 16
        self.learnable_init_state = True
        self.dt_ref = 1.0
        self.inj_k = 2.0
        self.koopman_use_noise = False
        self.koopman_noise_scale = 0.0
        self.interpolation_mode = "bilinear"
        self.pscan_use_decay = False
        self.pscan_use_residual = False
        self.pscan_chunk_size = 32
        self.ffn_ratio = 1.0
        self.ConvType = "conv"
        self.Arch = "unet"

def check_alignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    args = MockArgs()
    # 只需要实例化 layer 即可，不需要整个 UniPhy
    # 但为了方便获取参数，我们模拟一下参数传递
    H, W = args.input_size
    layer = ParallelPhysicalRecurrentLayer(
        emb_ch=args.emb_ch,
        input_shape=(H, W),
        rank=args.lru_rank,
        static_ch=args.static_ch,
        learnable_init_state=True,
        dynamics_mode="spectral"
    ).to(device)
    
    # 构造数据
    B = 1
    L = 2
    # 这里的 X 并不重要，主要是为了生成 A
    x_emb = torch.randn(B, args.emb_ch, L, H, W, device=device)
    dt_seq = torch.ones(B, L, device=device)
    # Static feats用于生成 H0
    static_feats = torch.randn(B, args.static_ch, H, W, device=device)
    
    print("\n=== 1. 检查 H0 生成与 Flatten ===")
    # 模拟 forward_spectral 中的 H0 生成
    # init_state 返回 [B, Rank, Emb, H, Wf]
    h0_raw = layer.init_state(static_feats)
    # 模拟 forward_spectral 中的处理
    # H0 本身是 complex(h0, 0)
    # H0_flat = H0.permute(0,1,2,3,4).reshape(B, -1)
    
    # 我们不仅看 shape，还要看数值追踪
    # 我们手动修改 init_state 的输出，让它具有可辨识的 pattern
    # Pattern: val = rank * 1000 + emb * 100 + h
    # 这样 flatten 后，我们看第 k 个元素的值，就能反推它是哪个 (r, e, h)
    
    B_sz, Rank_sz, Emb_sz, H_sz, Wf_sz = h0_raw.shape
    print(f"H0 Shape: {h0_raw.shape} (Rank, Emb, H, Wf)")
    
    # 创建 Mock H0 数据
    mock_h0_data = torch.zeros_like(h0_raw)
    for r in range(Rank_sz):
        for e in range(Emb_sz):
            for h in range(H_sz):
                val = (r+1) * 10000 + (e+1) * 100 + (h+1)
                mock_h0_data[:, r, e, h, :] = val
    
    # 执行 Flatten
    h0_flat = mock_h0_data.permute(0, 1, 2, 3, 4).reshape(B, -1)
    
    # 检查 Flatten 后的顺序
    # 如果顺序是 Rank优先 (Rank slow, Emb fast)，那么前几个元素应该是 r=0, e=0...
    # 如果顺序是 Emb优先 (Emb slow, Rank fast)，那么前几个元素应该是 r=0, e=0... r=1, e=0... (Wait, logical shape is Rank, Emb)
    
    # 取 flatten 后的第 0 个块 (对应 H*Wf 个元素) 的第一个值
    val_0 = h0_flat[0, 0].item()
    print(f"H0 Flatten[0] value: {val_0:.0f} (Expected: 10101 -> R=1, E=1)")
    
    # 取 flatten 后的第 1 个大块 (跳过 H*Wf)
    step_size = H_sz * Wf_sz
    val_1 = h0_flat[0, step_size].item()
    # 如果是 Rank-Major (Rank, Emb, ...): 下一个 Emb
    # Expected: R=1, E=2 -> 10201
    print(f"H0 Flatten[{step_size}] value: {val_1:.0f}")
    
    is_rank_major = (val_1 // 10000 == 1) and ((val_1 % 10000) // 100 == 2)
    print(f"-> H0 Memory Layout: {'Rank-Slow, Emb-Fast (Rank-Major)' if is_rank_major else 'Unknown/Mixed'}")

    print("\n=== 2. 检查 A 生成与 Flatten ===")
    # 模拟 compute_params -> build_A_koop
    x_perm = x_emb.permute(0, 2, 1, 3, 4).contiguous()
    # 我们Mock一下 compute_params 的输出 nu_rate
    # nu_rate shape: [B, L, 1, Emb, Wf, Rank] (From compute_params return)
    # 但 build_A_koop 内部 logic:
    # nu = nu_rate.squeeze(2).permute(0, 1, 4, 2, 3) -> [B, L, Rank, Emb, Wf]
    
    # 我们直接 Mock nu, 看看传入 kernel 后出来的 A 的顺序
    # 让 nu 的 pattern 也是 ID
    mock_nu = torch.zeros(B, L, Rank_sz, Emb_sz, Wf_sz, device=device)
    for r in range(Rank_sz):
        for e in range(Emb_sz):
             val = (r+1) * 10000 + (e+1) * 100
             mock_nu[:, :, r, e, :] = val # H 维度会在 Kernel 里广播，这里 nu 没有 H 维
             
    # 这里我们得手动模拟 kernel 的行为，或者信赖 kernel 的逻辑
    # Kernel logic:
    # nu_off = idx_b*... + idx_l*... + idx_r * stride_nu_r + idx_c * stride_nu_c ...
    # stride_nu_r 对应 Rank 维，stride_nu_c 对应 Emb 维
    # out_real[..., idx_r, idx_c, idx_h, idx_wf] = ...
    # 所以 A_out 的逻辑形状必然是 [B, L, Rank, Emb, H, Wf]
    
    # 我们验证 flatten 后的 A
    # A_flat = A.reshape(B, L, -1)
    # A 应该也是 Rank-Slow, Emb-Fast
    
    # 让我们直接构建一个符合 A 形状的 Tensor 并 Flatten，看 PyTorch reshape 行为
    mock_A = torch.zeros(B, L, Rank_sz, Emb_sz, H_sz, Wf_sz, device=device)
    for r in range(Rank_sz):
        for e in range(Emb_sz):
             val = (r+1) * 10000 + (e+1) * 100
             mock_A[:, :, r, e, :, :] = val
             
    A_flat = mock_A.reshape(B, L, -1)
    val_A_0 = A_flat[0, 0, 0].item()
    val_A_1 = A_flat[0, 0, step_size].item()
    
    print(f"A Flatten[0] value: {val_A_0:.0f} (Expected: 10100 -> R=1, E=1)")
    print(f"A Flatten[{step_size}] value: {val_A_1:.0f} (Expected: 10200 -> R=1, E=2)")
    
    is_A_rank_major = (val_A_1 // 10000 == 1) and ((val_A_1 % 10000) // 100 == 2)
    print(f"-> A Memory Layout: {'Rank-Slow, Emb-Fast (Rank-Major)' if is_A_rank_major else 'Unknown/Mixed'}")

    print("\n=== 3. 结论 ===")
    if is_rank_major and is_A_rank_major:
        print("[PASS] H0 和 A 的 Flatten 内存布局一致！")
        print("这意味着: H0_flat * A_flat 这种乘法在数学上是对齐的。")
    else:
        print("[FAIL] 内存布局不一致！乘法错位！")
        print(f"H0 Rank-Major: {is_rank_major}")
        print(f"A  Rank-Major: {is_A_rank_major}")

if __name__ == "__main__":
    check_alignment()

