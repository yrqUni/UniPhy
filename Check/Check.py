import os
import sys
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_dir = os.path.join(project_root, "Model", "ConvLRU")
sys.path.insert(0, model_dir)
sys.path.insert(0, project_root)

try:
    from ModelConvLRU import ConvLRU, RevINStats
except Exception:
    print(f"[Error] Could not import ConvLRU from {model_dir}")
    traceback.print_exc()
    raise


class MockArgs:
    def __init__(self, **kwargs):
        self.input_size = (32, 32)
        self.input_ch = 2
        self.out_ch = 2
        self.static_ch = 0
        self.hidden_factor = (2, 2)
        self.emb_ch = 16
        self.convlru_num_blocks = 2
        self.use_cbam = False
        self.lru_rank = 4
        self.Arch = "unet"
        self.down_mode = "avg"
        self.head_mode = "gaussian"
        self.ConvType = "conv"
        self.learnable_init_state = False
        self.ffn_ratio = 4.0
        self.sde_noise_mode = "none"
        for k, v in kwargs.items():
            setattr(self, k, v)


def slice_stats(stats: RevINStats, t0: int, t1: int) -> RevINStats:
    return RevINStats(mean=stats.mean[:, t0:t1], stdev=stats.stdev[:, t0:t1])


def check_tensor(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-6) -> None:
    if a.shape != b.shape:
        print(f"[WARN] {name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = (a - b).abs()
    max_diff = diff.max().item() if diff.numel() > 0 else 0.0
    mean_diff = diff.mean().item() if diff.numel() > 0 else 0.0
    if max_diff > atol:
        print(f"[FAIL] {name} max={max_diff:.6e} mean={mean_diff:.6e}")
    else:
        print(f"[PASS] {name} max={max_diff:.6e} mean={mean_diff:.6e}")


def _extract_out(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def make_slice_hook(acts: Dict[str, torch.Tensor], name: str, t_index: int) -> Any:
    def hook_fn(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        out = _extract_out(output)
        if not torch.is_tensor(out):
            return
        if out.dim() == 5:
            b, c, l, h, w = out.shape
            if l > 1 and 0 <= t_index < l:
                acts[name] = out[:, :, t_index].detach().clone()
            elif l == 1:
                acts[name] = out[:, :, 0].detach().clone()
            else:
                acts[name] = out.detach().clone()
        elif out.dim() == 4:
            acts[name] = out.detach().clone()
        else:
            acts[name] = out.detach().clone()
    return hook_fn


def debug_equivalence(model: ConvLRU) -> None:
    model.eval()
    device = next(model.parameters()).device

    b, l, c, h, w = 1, 5, 2, 32, 32
    torch.manual_seed(123)
    x = torch.randn(b, l, c, h, w, device=device)
    list_t = torch.ones(b, l, device=device)

    t_split = 3
    x_full = x[:, : t_split + 1]
    list_t_full = list_t[:, : t_split + 1]

    x_pre = x[:, :t_split]
    list_t_pre = list_t[:, :t_split]

    x_step = x[:, t_split : t_split + 1]
    list_t_step = list_t[:, t_split : t_split + 1]

    stats_full = model.revin.stats(x_full)
    stats_pre = slice_stats(stats_full, 0, t_split)
    stats_step = slice_stats(stats_full, t_split, t_split + 1)

    activations: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []

    hooks.append(model.embedding.register_forward_hook(make_slice_hook(activations, "Embedding", t_split)))
    for i, blk in enumerate(model.convlru_model.down_blocks):
        hooks.append(blk.register_forward_hook(make_slice_hook(activations, f"DownBlock_{i}", t_split)))
        if i < len(model.convlru_model.downsamples):
            hooks.append(model.convlru_model.downsamples[i].register_forward_hook(make_slice_hook(activations, f"DownSample_{i}", t_split)))
    hooks.append(model.convlru_model.mid_attention.register_forward_hook(make_slice_hook(activations, "MidAttention", t_split)))
    for i, blk in enumerate(model.convlru_model.up_blocks):
        hooks.append(blk.register_forward_hook(make_slice_hook(activations, f"UpBlock_{i}", t_split)))
    hooks.append(model.decoder.register_forward_hook(make_slice_hook(activations, "Decoder", t_split)))

    with torch.no_grad():
        _ = model(x_full, mode="p", listT=list_t_full, revin_stats=stats_full)

    for h_ in hooks:
        h_.remove()

    with torch.no_grad():
        _, states = model(x_pre, mode="p", listT=list_t_pre, revin_stats=stats_pre)

    with torch.no_grad():
        x_step_norm = model.revin.normalize(x_step, stats_step)
        x_emb, cond = model.embedding(x_step_norm, static_feats=None)

        curr_x = x_emb
        check_tensor(curr_x.squeeze(2), activations["Embedding"], "Embedding")

        num_down = len(model.convlru_model.down_blocks)
        skips: List[torch.Tensor] = []

        for i, blk in enumerate(model.convlru_model.down_blocks):
            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != curr_x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=curr_x.shape[-2:], mode="bilinear", align_corners=False)
            curr_x, _ = blk(curr_x, last_hidden_in=states[i], list_t=list_t_step, cond=curr_cond, static_feats=None)
            check_tensor(curr_x.squeeze(2), activations[f"DownBlock_{i}"], f"DownBlock_{i}")

            if i < len(model.convlru_model.down_blocks) - 1:
                skips.append(curr_x)
                x_s = curr_x
                if model.convlru_model.down_mode in {"shuffle", "avg", "conv"}:
                    pad_h = x_s.shape[-2] % 2
                    pad_w = x_s.shape[-1] % 2
                    if pad_h > 0 or pad_w > 0:
                        x_s = F.pad(x_s, (0, pad_w, 0, pad_h))
                if x_s.shape[-2] >= 2 and x_s.shape[-1] >= 2:
                    curr_x = model.convlru_model.downsamples[i](x_s)
                else:
                    curr_x = x_s
                check_tensor(curr_x.squeeze(2), activations[f"DownSample_{i}"], f"DownSample_{i}")

        curr_x = model.convlru_model.mid_attention(curr_x)
        check_tensor(curr_x.squeeze(2), activations["MidAttention"], "MidAttention")

        for i, blk in enumerate(model.convlru_model.up_blocks):
            curr_x = model.convlru_model.upsample(curr_x)
            skip = skips.pop()
            if curr_x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.size(-2) - curr_x.size(-2)
                diff_x = skip.size(-1) - curr_x.size(-1)
                curr_x = F.pad(curr_x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

            if model.convlru_model.arch_mode == "bifpn":
                curr_x = model.convlru_model.csa_blocks[i](curr_x, skip)
            else:
                skip2 = model.convlru_model.csa_blocks[i](skip, curr_x)
                curr_x = torch.cat([curr_x, skip2], dim=1)
                curr_x = model.convlru_model.fusion(curr_x)

            curr_cond = cond
            if curr_cond is not None and curr_cond.shape[-2:] != curr_x.shape[-2:]:
                curr_cond = F.interpolate(curr_cond, size=curr_x.shape[-2:], mode="bilinear", align_corners=False)

            curr_x, _ = blk(curr_x, last_hidden_in=states[num_down + i], list_t=list_t_step, cond=curr_cond, static_feats=None)
            check_tensor(curr_x.squeeze(2), activations[f"UpBlock_{i}"], f"UpBlock_{i}")

        out = model.decoder(curr_x, cond=cond, timestep=None)
        step_dec = out.squeeze(2)
        full_dec = activations["Decoder"]
        check_tensor(step_dec, full_dec, "Decoder")


def main() -> None:
    set_deterministic(42)
    args = MockArgs(Arch="unet", input_size=(32, 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLRU(args).to(device)
    debug_equivalence(model)


if __name__ == "__main__":
    main()

