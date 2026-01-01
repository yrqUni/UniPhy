import os
import sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "Model", "ConvLRU")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from ModelConvLRU import ConvLRU

class MockArgs:
    def __init__(self):
        self.input_ch = 4
        self.out_ch = 4
        self.input_size = (64, 64)
        self.emb_ch = 32
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.static_ch = 2
        self.hidden_factor = (2, 2)
        self.convlru_num_blocks = 2
        self.ffn_hidden_ch = 64
        self.ffn_hidden_layers_num = 1
        self.use_cbam = False
        self.num_expert = 4
        self.activate_expert = 2
        self.lru_rank = 8
        self.use_selective = True
        self.use_gate = True
        self.use_freq_prior = False
        self.use_sh_prior = True
        self.sh_Lmax = 4
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.head_mode = "gaussian"
        self.dec_hidden_ch = 32
        self.dec_hidden_layers_num = 0
        self.dec_strategy = "pxsf"
        self.unet = True
        self.down_mode = "shuffle"

def check_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    args = MockArgs()
    model = ConvLRU(args).to(device)
    model.eval()

    for name, module in model.named_modules():
        if hasattr(module, "forcing_scale") and isinstance(module.forcing_scale, torch.nn.Parameter):
            module.forcing_scale.data.fill_(0.0)

    B, L, H, W = 2, 8, args.input_size[0], args.input_size[1]
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    static = torch.randn(B, args.static_ch, H, W, device=device)
    listT = torch.rand(B, L, device=device)

    with torch.no_grad():
        out_p = model(x, mode="p", listT=listT, static_feats=static)

    outputs_i = []
    
    with torch.no_grad():
        cond = None
        if static is not None and getattr(model.embedding, "static_ch", 0) > 0 and model.embedding.static_embed is not None:
            cond = model.embedding.static_embed(static)

        last_hidden_ins = None

        for t in range(L):
            x_t = x[:, t:t+1, ...]
            dt_t = listT[:, t:t+1]

            x_t_norm = model.revin(x_t, "norm")
            x_t_emb, _ = model.embedding(x_t_norm, static_feats=static)

            x_t_hid, last_hidden_ins = model.convlru_model(
                x_t_emb,
                last_hidden_ins=last_hidden_ins,
                listT=dt_t,
                cond=cond,
            )

            out_t = model.decoder(x_t_hid, cond=cond, timestep=None)
            out_t = out_t.permute(0, 2, 1, 3, 4).contiguous()

            if model.decoder.head_mode == "gaussian":
                mu, sigma = torch.chunk(out_t, 2, dim=2)
                if mu.size(2) == model.revin.num_features:
                    mu = model.revin(mu, "denorm")
                    sigma = sigma * model.revin.stdev
                out_step = torch.cat([mu, sigma], dim=2)
            else:
                if out_t.size(2) == model.revin.num_features:
                    out_step = model.revin(out_t, "denorm")
                else:
                    out_step = out_t

            outputs_i.append(out_step)

    out_i = torch.cat(outputs_i, dim=1)

    diff = torch.abs(out_p - out_i)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max Absolute Difference: {max_diff:.6e}")
    print(f"Mean Absolute Difference: {mean_diff:.6e}")

    threshold = 1e-4

    if max_diff < threshold:
        print("SUCCESS")
    else:
        print("FAILURE")

if __name__ == "__main__":
    check_equivalence()

