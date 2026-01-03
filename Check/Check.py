import sys
import os
import torch
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
model_dir = os.path.join(project_root, 'Model', 'ConvLRU')
sys.path.insert(0, model_dir)
sys.path.insert(0, project_root)

try:
    from ModelConvLRU import ConvLRU
except ImportError:
    print(f"[Error] Could not import ConvLRU from {model_dir}")
    sys.exit(1)

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
        self.num_expert = 1
        self.activate_expert = 1
        self.lru_rank = 4
        self.Arch = "unet"
        self.down_mode = "avg"
        self.head_mode = "gaussian"
        self.use_checkpointing = False
        self.ConvType = "conv"
        self.use_spectral_mixing = False
        self.use_advection = False
        self.use_spatial_ssm = True
        self.use_stochastic = False
        self.learnable_init_state = False
        self.use_wavelet_ssm = False
        self.use_cross_var_attn = False
        self.use_freq_prior = False
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0
        self.use_graph_interaction = False
        self.ffn_ratio = 4.0
        self.loss = ["lat"]
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_case(case_name, args_dict, mode='p'):
    print(f"[{case_name}] Setting up...")
    try:
        args = MockArgs(**args_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvLRU(args).to(device)
        
        H, W = args.input_size
        B, L, C = 2, 4, args.input_ch
        
        if "Full_ERA5" in case_name:
            B = 1
            
        x = torch.randn(B, L, C, H, W).to(device)
        static_feats = None
        if args.static_ch > 0:
            static_feats = torch.randn(B, args.static_ch, H, W).to(device)
        
        timestep = None
        if args.head_mode in ["diffusion", "flow"]:
            timestep = torch.randint(0, 5, (B,)).to(device)
            
        listT = torch.ones(B, L).to(device)

        if mode == 'p':
            out = model(x, mode='p', listT=listT, static_feats=static_feats, timestep=timestep)
            if isinstance(out, tuple):
                out = out[0]
            print(f"[{case_name}] Success. Output: {out.shape}")
            
        elif mode == 'i':
            out_gen_num = 3
            listT_future = torch.ones(B, out_gen_num - 1).to(device)
            out = model(x, mode='i', out_gen_num=out_gen_num, listT=listT, listT_future=listT_future, static_feats=static_feats, timestep=timestep)
            print(f"[{case_name}] Inference Success. Output: {out.shape}")

    except Exception as e:
        print(f"[{case_name}] Failed: {e}")
        traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        if 'x' in locals(): del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("========== Check Start ==========")
    run_case("01_HKLF_Baseline", {})
    run_case("02_Spatial_SSM", {"use_spatial_ssm": True})
    run_case("03_Stochastic", {"use_stochastic": True})
    run_case("04_Advection_Lie", {"use_advection": True})
    run_case("05_Spectral_SH", {"use_spectral_mixing": True, "use_freq_prior": True, "use_sh_prior": True})
    run_case("06_Wavelet_CrossVar", {"use_wavelet_ssm": True, "use_cross_var_attn": True})
    run_case("07_Graph_Interaction", {"use_graph_interaction": True})
    run_case("08_BiFPN", {"Arch": "bifpn"})
    run_case("09_No_UNet", {"Arch": "no_unet"})
    run_case("10_MoE", {"num_expert": 4, "activate_expert": 2})
    run_case("11_Diffusion_Head", {"head_mode": "diffusion", "out_ch": 2})
    run_case("12_Token_Head", {"head_mode": "token", "out_ch": 2})
    run_case("13_Static_Init", {"static_ch": 4, "learnable_init_state": True})
    run_case("14_Inference", {"use_spatial_ssm": True}, mode='i')
    run_case("15_Full_ERA5_Size", {"input_size": (721, 1440), "input_ch": 2, "emb_ch": 16, "hidden_factor": (7, 12)}, mode='p')
    print("========== Check Done ==========")

if __name__ == "__main__":
    main()

