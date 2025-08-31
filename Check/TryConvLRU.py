import os, sys, gc, math, time, itertools, random
import torch
import numpy as np

sys.path.append('../Model/ConvLRU/')
from ModelConvLRU import ConvLRU
from pscan import pscan_check

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(42)

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device('cpu')
)

class Args:
    def __init__(self,
                 input_size=(144,288),
                 hidden_factor=(6,12),
                 emb_strategy='pxus',
                 dec_strategy='pxsf',
                 use_cbam=False,
                 use_gate=True,
                 use_freq_prior=False,
                 use_sh_prior=False,
                 freq_rank=8,
                 freq_gain_init=0.0,
                 sh_Lmax=6,
                 sh_rank=8,
                 sh_gain_init=0.0,
                 lru_rank=32,
                 emb_ch=32,
                 convlru_num_blocks=1,
                 emb_hidden_ch=32,
                 emb_hidden_layers_num=1,
                 ffn_hidden_ch=32,
                 ffn_hidden_layers_num=1,
                 dec_hidden_ch=0,
                 dec_hidden_layers_num=0,
                 input_ch=8,
                 out_ch=8,
                 gen_factor=4,
                 hidden_activation='ReLU',
                 output_activation='Sigmoid'):
        self.sample_input_size = input_size
        self.input_size = input_size
        self.input_ch = input_ch
        self.out_ch = out_ch
        self.emb_ch = emb_ch
        self.convlru_num_blocks = convlru_num_blocks
        self.hidden_factor = hidden_factor
        self.use_gate = use_gate
        self.emb_hidden_ch = emb_hidden_ch
        self.emb_hidden_layers_num = emb_hidden_layers_num
        self.emb_strategy = emb_strategy
        self.ffn_hidden_ch = ffn_hidden_ch
        self.ffn_hidden_layers_num = ffn_hidden_layers_num
        self.use_cbam = use_cbam
        self.dec_hidden_ch = dec_hidden_ch
        self.dec_hidden_layers_num = dec_hidden_layers_num
        self.dec_strategy = dec_strategy
        self.gen_factor = gen_factor
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_freq_prior = use_freq_prior
        self.freq_rank = freq_rank
        self.freq_gain_init = freq_gain_init
        self.use_sh_prior = use_sh_prior
        self.sh_Lmax = sh_Lmax
        self.sh_rank = sh_rank
        self.sh_gain_init = sh_gain_init
        self.lru_rank = lru_rank

def check_finite(*tensors):
    for t in tensors:
        assert torch.isfinite(t).all()

def run_once(name, args, B=1, L=8, out_frames_num=8, lr=1e-3):
    t0 = time.time()
    print(f"[case]{name}")
    print(f"[dev]{device}")
    model = ConvLRU(args).to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params]{total_params} {trainable_params}")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    H, W = args.sample_input_size
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    y = torch.randn(B, L, args.out_ch, H, W, device=device)
    opt.zero_grad(set_to_none=True)
    out_p = model(x, mode='p')
    assert out_p.shape == y.shape
    check_finite(out_p)
    loss_p = loss_fn(out_p, y)
    check_finite(loss_p)
    loss_p.backward()
    opt.step()
    print(f"[p]loss {loss_p.item():.6f} shape {tuple(out_p.shape)}")
    del y, out_p, loss_p
    if args.input_ch == args.out_ch:
        assert out_frames_num % args.gen_factor == 0
        assert L >= args.gen_factor
        y_i = torch.randn(B, out_frames_num, args.out_ch, H, W, device=device)
        out_gen_num = out_frames_num // args.gen_factor
        opt.zero_grad(set_to_none=True)
        out_i = model(x, mode='i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
        assert out_i.shape == y_i.shape
        check_finite(out_i)
        loss_i = loss_fn(out_i, y_i)
        check_finite(loss_i)
        loss_i.backward()
        opt.step()
        print(f"[i]loss {loss_i.item():.6f} shape {tuple(out_i.shape)}")
        del y_i, out_i, loss_i
    del x, model, loss_fn, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    print(f"[time]{time.time()-t0:.3f}s")

def main():
    print(f"[pscan]{pscan_check()}")
    base_B = 1
    base_L = 8
    base_frames = 8
    configs = []
    configs.append(("square_pxus_pxsf_no_priors",
                    Args(input_size=(144,144), hidden_factor=(6,6),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         use_cbam=False, use_freq_prior=False, use_sh_prior=False,
                         gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("rect_pxus_pxsf_no_priors",
                    Args(input_size=(144,288), hidden_factor=(6,12),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         use_cbam=False, use_freq_prior=False, use_sh_prior=False,
                         gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("rect_pxus_deconv_no_priors",
                    Args(input_size=(144,288), hidden_factor=(6,12),
                         emb_strategy='pxus', dec_strategy='deconv',
                         use_cbam=False, use_freq_prior=False, use_sh_prior=False,
                         gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("square_conv_pxsf_cbam",
                    Args(input_size=(144,144), hidden_factor=(6,6),
                         emb_strategy='conv', dec_strategy='pxsf',
                         use_cbam=True, use_freq_prior=False, use_sh_prior=False,
                         gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("rect_pxus_pxsf_freqprior",
                    Args(input_size=(144,288), hidden_factor=(6,12),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         use_cbam=False, use_freq_prior=True, use_sh_prior=False,
                         freq_rank=8, gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("square_pxus_pxsf_shprior",
                    Args(input_size=(144,144), hidden_factor=(6,6),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         use_cbam=False, use_freq_prior=False, use_sh_prior=True,
                         sh_Lmax=6, gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("rect_pxus_pxsf_both_priors",
                    Args(input_size=(144,288), hidden_factor=(6,12),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         use_cbam=False, use_freq_prior=True, use_sh_prior=True,
                         freq_rank=8, sh_Lmax=6, gen_factor=4, input_ch=8, out_ch=8)))
    configs.append(("full_720x1440_light",
                    Args(input_size=(720,1440), hidden_factor=(10,20),
                         emb_strategy='pxus', dec_strategy='pxsf',
                         emb_ch=16, emb_hidden_ch=16, ffn_hidden_ch=16,
                         convlru_num_blocks=1, emb_hidden_layers_num=1, ffn_hidden_layers_num=1,
                         use_cbam=False, use_freq_prior=False, use_sh_prior=False,
                         gen_factor=2, input_ch=4, out_ch=4)))
    for name, a in configs:
        L = base_L if min(a.input_size) <= 288 else 2
        frames = base_frames if min(a.input_size) <= 288 else a.gen_factor * 2
        run_once(name, a, B=base_B, L=L, out_frames_num=frames, lr=1e-3)

if __name__ == "__main__":
    main()
