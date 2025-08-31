import sys, math, gc, torch
sys.path.append('../Model/ConvLRU/')
from ModelConvLRU import ConvLRU

def device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

class Args:
    def __init__(self, prior_mode='fft_dct'):
        self.sample_input_size = (720, 1440)
        self.input_size = (720, 1440)
        self.input_ch = 20
        self.out_ch = 20
        self.emb_ch = 32
        self.convlru_num_blocks = 1
        self.hidden_factor = (10, 20)
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.emb_strategy = 'pxus'
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.dec_strategy = 'pxsf'
        self.gen_factor = 8
        self.hidden_activation = 'ReLU'
        self.output_activation = 'Sigmoid'
        self.prior_mode = prior_mode
        self.orth_every = 50
        self.state_size = 64
        self.ms_enable = True
        self.ms_scale = 2
        self.ms_state_size = 32
        self.gate_lru = True
        self.gate_conv = True
        self.use_cbam = False
        self.cbam_reduction = 16
        self.cbam_kernel = (1, 7, 7)
        self.mix_ratio = 4
        self.mix_groups = 1
        self.mix_act = 'SiLU'

def run_once(prior_mode):
    d = device()
    args = Args(prior_mode=prior_mode)
    B, L = 2, 8
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = ConvLRU(args).to(d)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{prior_mode}] GP: {total_params}  TP: {trainable_params}")

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(B, L, args.input_ch, *args.sample_input_size, device=d)
    y = torch.randn(B, L, args.out_ch, *args.sample_input_size, device=d)
    opt.zero_grad()
    yp = model(x, mode='p')
    loss_p = loss_fn(yp, y)
    loss_p.backward()
    opt.step()
    print(f"[{prior_mode}] p mode Loss {loss_p.item():.6f}")

    del x, y, yp, loss_p
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model.train()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    out_frames_num = 8
    x = torch.randn(B, L, args.input_ch, *args.sample_input_size, device=d)
    y = torch.randn(B, out_frames_num, args.out_ch, *args.sample_input_size, device=d)
    opt.zero_grad()
    out_gen_num = max(1, (out_frames_num + args.gen_factor - 1) // args.gen_factor)
    yi = model(x, mode='i', out_gen_num=out_gen_num, gen_factor=args.gen_factor)
    yi = yi[:, :out_frames_num]
    loss_i = loss_fn(yi, y)
    loss_i.backward()
    opt.step()
    print(f"[{prior_mode}] i mode Loss {loss_i.item():.6f}")

    del model, loss_fn, opt, x, y, yi, loss_i
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_once('fft_dct')
    run_once('sph_harm')
