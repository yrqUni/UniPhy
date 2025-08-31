import sys, os, gc, math
sys.path.append('../Model/ConvLRU/')

import torch
from ModelConvLRU import ConvLRU

torch.manual_seed(42)
device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f"{device}")

class Args:
    def __init__(self,
                 input_size=(720, 1440),
                 hidden_factor=(10, 20),
                 emb_strategy='pxus',
                 dec_strategy='pxsf',
                 use_cbam=False,
                 use_gate=True,
                 use_freq_prior=False,
                 use_sh_prior=False,
                 gen_factor=8):
        self.sample_input_size = input_size
        self.input_size = input_size
        self.input_ch = 20
        self.out_ch = 20
        self.emb_ch = 32
        self.convlru_num_blocks = 1
        self.hidden_factor = hidden_factor
        self.use_gate = use_gate
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.emb_strategy = emb_strategy
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.use_cbam = use_cbam
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.dec_strategy = dec_strategy
        self.gen_factor = gen_factor
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Sigmoid'
        self.use_freq_prior = use_freq_prior
        self.freq_rank = 8
        self.freq_gain_init = 0.0
        self.use_sh_prior = use_sh_prior
        self.sh_Lmax = 6
        self.sh_rank = 8
        self.sh_gain_init = 0.0
        self.lru_rank = 32

def run_once(args, B=1, L=4, out_frames_num=8, mode='p'):
    model = ConvLRU(args).to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params} {trainable_params}")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    H, W = args.sample_input_size
    x = torch.randn(B, L, args.input_ch, H, W, device=device)
    if mode == 'p':
        y = torch.randn(B, L, args.out_ch, H, W, device=device)
        opt.zero_grad(set_to_none=True)
        out = model(x, mode='p')
        assert out.shape == y.shape
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print(f"{loss.item():.6f} {tuple(out.shape)}")
    else:
        assert args.input_ch == args.out_ch
        assert out_frames_num % args.gen_factor == 0
        assert L >= args.gen_factor
        y = torch.randn(B, out_frames_num, args.out_ch, H, W, device=device)
        out_gen_num = out_frames_num // args.gen_factor
        opt.zero_grad(set_to_none=True)
        out = model(x, mode='i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
        assert out.shape == y.shape
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print(f"{loss.item():.6f} {tuple(out.shape)}")
    del model, loss_fn, opt, x, y, out
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

print("\nSMOKE")
args_smoke = Args(
    input_size=(144, 288),
    hidden_factor=(6, 12),
    emb_strategy='pxus',
    dec_strategy='pxsf',
    use_cbam=False,
    use_gate=True,
    use_freq_prior=False,
    use_sh_prior=False,
    gen_factor=8
)
run_once(args_smoke, B=1, L=8, out_frames_num=8, mode='p')
run_once(args_smoke, B=1, L=8, out_frames_num=8, mode='i')

print("\nFULL")
args_full = Args(
    input_size=(720, 1440),
    hidden_factor=(10, 20),
    emb_strategy='pxus',
    dec_strategy='pxsf',
    use_cbam=False,
    use_gate=True,
    use_freq_prior=False,
    use_sh_prior=False,
    gen_factor=2
)
run_once(args_full, B=1, L=2, out_frames_num=8, mode='p')
run_once(args_full, B=1, L=2, out_frames_num=8, mode='i')
