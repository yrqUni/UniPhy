import os
import sys
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU, RevINStats


def _env_info(device: torch.device) -> str:
    amp = "OFF"
    dtype = torch.float32
    det = torch.are_deterministic_algorithms_enabled()
    det_mode = "on" if det else "warn"
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)
    s = f"[INFO] Device={device.type} AMP={amp} dtype={dtype} determinism={det_mode}"
    if cublas is not None:
        s += f"\n[INFO] CUBLAS_WORKSPACE_CONFIG={cublas}"
    return s


def _make_args() -> Any:
    return SimpleNamespace(
        input_ch=2,
        out_ch=1,
        input_size=(64, 64),
        emb_ch=32,
        static_ch=0,
        hidden_factor=(2, 2),
        ConvType="conv",
        Arch="unet",
        head_mode="gaussian",
        convlru_num_blocks=2,
        down_mode="avg",
        use_cbam=False,
        ffn_ratio=4.0,
        lru_rank=32,
        koopman_use_noise=False,
        koopman_noise_scale=1.0,
        learnable_init_state=False,
    )


def _max_mean(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def _path_check(model: ConvLRU, x_full: torch.Tensor, listT_full: torch.Tensor, static_feats: Optional[torch.Tensor]) -> None:
    model.eval()
    with torch.no_grad():
        _ = model(x_full, mode="p", listT=listT_full, static_feats=static_feats)
        _ = model(x_full, mode="p", listT=None, static_feats=static_feats)
        _ = model(x_full, mode="p", listT=listT_full, static_feats=None)
        st = model.revin.stats(x_full)
        _ = model.revin(model.revin(x_full, "norm", stats=st), "denorm", stats=st)
        _ = model(x_full, mode="p", listT=listT_full, static_feats=static_feats, revin_stats=st)
    print("[PASS] p: listT + static_feats")
    print("[PASS] p: listT=None + static_feats")
    print("[PASS] p: static_feats=None")
    print("[PASS] revin: direct norm+denorm")
    print("[PASS] p: explicit revin_stats")


def _teacher_forcing_numeric(model: ConvLRU, x_full: torch.Tensor, listT_full: torch.Tensor, static_feats: Optional[torch.Tensor], tol_max: float = 1e-5, tol_mean: float = 1e-6) -> None:
    model.eval()
    with torch.no_grad():
        stats_full = model.revin.stats(x_full)
        y_p, _ = model(x_full, mode="p", listT=listT_full, static_feats=static_feats, revin_stats=stats_full)

        B, L, C, H, W = x_full.shape
        head_mode = str(getattr(model.decoder, "head_mode", "gaussian")).lower()

        cond = None
        if model.embedding.static_ch > 0 and model.embedding.static_embed is not None and static_feats is not None:
            cond = model.embedding.static_embed(static_feats)

        last_hidden = None
        outs = []
        for t in range(L):
            x_t = x_full[:, t : t + 1]
            dt_t = listT_full[:, t : t + 1] if listT_full is not None else None
            st_t = RevINStats(mean=stats_full.mean[:, t : t + 1], stdev=stats_full.stdev[:, t : t + 1])

            x_norm = model.revin(x_t, "norm", stats=st_t)
            x_emb, _ = model.embedding(x_norm, static_feats=static_feats)
            x_hid, last_hidden = model.convlru_model(x_emb, last_hidden_ins=last_hidden, listT=dt_t, cond=cond, static_feats=static_feats)
            out = model.decoder(x_hid, cond=cond, timestep=None)
            out_t = out.permute(0, 2, 1, 3, 4).contiguous()

            if head_mode == "gaussian":
                mu, sigma = torch.chunk(out_t, 2, dim=2)
                if mu.size(2) == model.revin.num_features:
                    mu = model.revin(mu, "denorm", stats=st_t)
                    sigma = sigma * st_t.stdev
                outs.append(torch.cat([mu, sigma], dim=2))
            else:
                if out_t.size(2) == model.revin.num_features:
                    out_t = model.revin(out_t, "denorm", stats=st_t)
                outs.append(out_t)

        y_tf = torch.cat(outs, dim=1)

        if tuple(y_tf.shape) != tuple(y_p.shape):
            raise RuntimeError(f"shape mismatch p={tuple(y_p.shape)} tf={tuple(y_tf.shape)}")

        mx, mn = _max_mean(y_tf, y_p)
        print(f"[RESULT] TF(GT stepwise) vs p(full) max={mx:.6e} mean={mn:.6e}")

        ok = (mx <= tol_max) and (mn <= tol_mean)
        for t in range(L):
            mx_t, mn_t = _max_mean(y_tf[:, t : t + 1], y_p[:, t : t + 1])
            print(f"[STEP {t}] max={mx_t:.6e} mean={mn_t:.6e}")
            if mx_t > tol_max or mn_t > tol_mean:
                ok = False

        if not ok:
            raise RuntimeError("numeric mismatch")


def main() -> None:
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(_env_info(device))

    args = _make_args()
    model = ConvLRU(args).to(device).eval()

    B = 1
    L = 6
    C = int(getattr(args, "input_ch", 2))
    H, W = tuple(getattr(args, "input_size", (64, 64)))

    x_full = torch.randn(B, L, C, H, W, device=device, dtype=torch.float32)
    listT_full = torch.ones(B, L, device=device, dtype=torch.float32)
    static_feats = None
    if int(getattr(args, "static_ch", 0)) > 0:
        static_feats = torch.randn(B, int(args.static_ch), H, W, device=device, dtype=torch.float32)

    print("\n[PATH CHECK]")
    _path_check(model, x_full, listT_full, static_feats)

    print("\n[NUMERIC CONSISTENCY]")
    _teacher_forcing_numeric(model, x_full, listT_full, static_feats)


if __name__ == "__main__":
    main()

