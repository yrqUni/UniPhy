import torch
import sys
import os
import traceback
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Model/ConvLRU")))

try:
    from ModelConvLRU import ConvLRU
except ImportError:
    if os.path.exists("ModelConvLRU.py"):
        from ModelConvLRU import ConvLRU
    else:
        print("Error: Could not import ModelConvLRU. Please check python path.")
        sys.exit(1)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MockArgs:
    def __init__(self, **kwargs):
        self.input_size = (32, 32)
        self.input_ch = 4
        self.out_ch = 3 
        self.emb_ch = 16
        self.emb_hidden_ch = 32
        self.emb_hidden_layers_num = 1
        self.convlru_num_blocks = 2
        self.lru_rank = 8
        self.use_selective = True
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.num_expert = -1
        self.activate_expert = 2
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 32
        self.dec_hidden_layers_num = 1
        self.static_ch = 0
        self.head_mode = "gaussian"
        self.hidden_factor = (2, 2)
        self.Arch = "unet"
        self.down_mode = "shuffle"
        self.use_cbam = False
        self.use_gate = False
        self.use_spectral_mixing = False
        self.use_anisotropic_diffusion = False
        self.use_advection = False
        self.use_graph_interaction = False
        self.use_adaptive_ssm = False
        self.use_neural_operator = False
        self.learnable_init_state = False
        self.use_wavelet_ssm = False
        self.use_cross_var_attn = False
        self.ConvType = "conv"
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_test_case(test_name, args):
    print(f"Testing Path: [{test_name}] ... ", end="", flush=True)
    
    try:
        model = ConvLRU(args).cuda()
        B, L, C, H, W = 2, 4, args.input_ch, args.input_size[0], args.input_size[1]
        x = torch.randn(B, L, C, H, W).cuda()
        
        static_feats = None
        if args.static_ch > 0:
            static_feats = torch.randn(B, args.static_ch, H, W).cuda()
            
        timestep = None
        if args.head_mode in ["diffusion", "flow"]:
            timestep = torch.randint(0, 1000, (B,)).cuda()

        out_p = model(x, mode="p", static_feats=static_feats, timestep=timestep)
        
        expected_out_ch = args.out_ch * 2 if args.head_mode == "gaussian" else args.out_ch
        
        if args.head_mode != "token":
             assert out_p.shape == (B, L, expected_out_ch, H, W), \
                 f"P-Mode Shape Mismatch: Expected {(B, L, expected_out_ch, H, W)}, got {out_p.shape}"
        else:
             assert isinstance(out_p, tuple)
             assert out_p[0].shape == (B, L, expected_out_ch, H, W)

        cond_len = 2
        pred_len = 2
        x_cond = x[:, :cond_len]
        listT_cond = torch.ones(B, cond_len).cuda()
        listT_future = torch.ones(B, pred_len).cuda()
        
        out_i = model(
            x_cond, 
            mode="i", 
            out_gen_num=pred_len, 
            listT=listT_cond, 
            listT_future=listT_future,
            static_feats=static_feats, 
            timestep=timestep
        )
        
        assert out_i.shape == (B, pred_len, expected_out_ch, H, W), \
            f"I-Mode Shape Mismatch: Expected {(B, pred_len, expected_out_ch, H, W)}, got {out_i.shape}"

        print("PASSED")
        
        del model, x, out_p, out_i, static_feats, timestep
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print("FAILED")
        traceback.print_exc()
        return False

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is required for this check.")
        return

    set_seed(1234)
    print("=== ConvLRU Full Path Coverage Check ===\n")

    args_baseline = MockArgs()
    run_test_case("Baseline (UNet + Gaussian)", args_baseline)

    args_bifpn = MockArgs(Arch="bifpn")
    run_test_case("Arch: BiFPN", args_bifpn)

    args_no_unet = MockArgs(Arch="no_unet")
    run_test_case("Arch: No UNet", args_no_unet)

    args_advanced = MockArgs(
        use_spectral_mixing=True,
        use_anisotropic_diffusion=True,
        use_advection=True,
        use_graph_interaction=True,
        use_adaptive_ssm=True,
        use_neural_operator=True,
        use_wavelet_ssm=True,
        use_cross_var_attn=True,
        use_gate=True,
        use_cbam=True,
        use_freq_prior=True,
        use_sh_prior=True
    )
    run_test_case("All Advanced Modules", args_advanced)

    args_moe = MockArgs(num_expert=4, activate_expert=2)
    run_test_case("Mixture of Experts (MoE)", args_moe)

    args_diff = MockArgs(head_mode="diffusion", out_ch=3, dec_hidden_layers_num=1)
    run_test_case("Head: Diffusion", args_diff)

    args_token = MockArgs(head_mode="token", out_ch=16) 
    run_test_case("Head: Token (VQ)", args_token)

    args_misc = MockArgs(
        down_mode="conv",
        dec_strategy="deconv",
        static_ch=4,
        learnable_init_state=True
    )
    run_test_case("Misc: ConvDown + Deconv + Static", args_misc)

    args_avg = MockArgs(down_mode="avg")
    run_test_case("Misc: Avg Downsample", args_avg)

    print("\n=== Check Complete ===")

if __name__ == "__main__":
    main()

