import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model', 'ConvLRU')))

try:
    from ModelConvLRU import ConvLRU
except ImportError:
    # Fallback if running from a different root
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Model', 'ConvLRU')))
    from ModelConvLRU import ConvLRU

class MockArgs:
    def __init__(self, **kwargs):
        self.input_size = (32, 32)
        self.input_ch = 2
        self.out_ch = 2
        self.static_ch = 0
        self.hidden_activation = "SiLU"
        self.output_activation = "Tanh"
        self.emb_strategy = "pxus"
        self.hidden_factor = (2, 2)
        self.emb_ch = 16
        self.emb_hidden_ch = 16
        self.emb_hidden_layers_num = 1
        self.convlru_num_blocks = 2
        self.use_cbam = False
        self.ffn_hidden_ch = 32
        self.ffn_hidden_layers_num = 1
        self.num_expert = 1
        self.activate_expert = 1
        self.use_gate = False
        self.lru_rank = 4
        self.use_selective = False
        self.unet = True
        self.down_mode = "avg"
        self.use_freq_prior = False
        self.freq_rank = 4
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.head_mode = "gaussian"
        self.diffusion_steps = 5
        self.use_checkpointing = False
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
        self.Arch = "unet"
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_case(case_name, args_dict, input_shape=(2, 4, 2, 32, 32), mode='p'):
    print(f"[{case_name}] Setting up...")
    args = MockArgs(**args_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = ConvLRU(args).to(device)
    except Exception as e:
        print(f"[{case_name}] Initialization Failed: {e}")
        return

    B, L, C, H, W = input_shape
    x = torch.randn(B, L, C, H, W).to(device)
    
    static_feats = None
    if args.static_ch > 0:
        static_feats = torch.randn(B, args.static_ch, H, W).to(device)
        
    listT = torch.ones(B, L).to(device)
    
    timestep = None
    if args.head_mode in ["diffusion", "flow"]:
        timestep = torch.randint(0, 5, (B,)).to(device)
        
    try:
        if mode == 'p':
            out = model(x, mode='p', listT=listT, static_feats=static_feats, timestep=timestep)
            if isinstance(out, tuple):
                print(f"[{case_name}] Success. Output: tuple of length {len(out)}")
            else:
                print(f"[{case_name}] Success. Output shape: {out.shape}")
        elif mode == 'i':
            out_gen_num = 3
            listT_future = torch.ones(B, out_gen_num - 1).to(device)
            out = model(x, mode='i', out_gen_num=out_gen_num, listT=listT, listT_future=listT_future, static_feats=static_feats, timestep=timestep)
            print(f"[{case_name}] Inference Success. Output shape: {out.shape}")
            
    except Exception as e:
        print(f"[{case_name}] Forward Execution Failed: {e}")
        import traceback
        traceback.print_exc()
    
    del model, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print("========== Start Checking ConvLRU Model Paths ==========")
    
    run_case("01_Standard_UNet_Gaussian", {})
    
    run_case("02_No_UNet", {"Arch": "no_unet", "convlru_num_blocks": 2})
    
    run_case("03_BiFPN", {"Arch": "bifpn"})
    
    run_case("04_Downsample_Conv", {"down_mode": "conv"})
    run_case("05_Downsample_Shuffle", {"down_mode": "shuffle"})
    
    run_case("06_Advanced_LRU_Components_1", {
        "use_selective": True,
        "use_spectral_mixing": True,
        "use_anisotropic_diffusion": True, 
        "use_advection": True,
        "use_graph_interaction": True
    })
    
    run_case("07_Advanced_LRU_Components_2", {
        "use_adaptive_ssm": True,
        "use_neural_operator": True,
        "use_wavelet_ssm": True,
        "use_cross_var_attn": True
    })
    
    run_case("08_Priors_and_Gates", {
        "use_freq_prior": True,
        "use_sh_prior": True,
        "use_gate": True
    })
    
    run_case("09_Static_Features_Init", {
        "static_ch": 4,
        "learnable_init_state": True
    })
    
    run_case("10_Static_Features_Embedding_Cond", {
        "static_ch": 4,
        "emb_hidden_layers_num": 1
    })

    run_case("11_MoE_Structure", {
        "num_expert": 4,
        "activate_expert": 2,
        "ffn_hidden_ch": 64
    })
    
    run_case("12_ConvType_DCN", {
        "ConvType": "dcn"
    })
    
    run_case("13_Use_CBAM", {
        "use_cbam": True
    })
    
    run_case("14_Head_Diffusion", {
        "head_mode": "diffusion", 
        "out_ch": 2
    })
    
    run_case("15_Head_Token", {
        "head_mode": "token",
        "out_ch": 2 
    })
    
    run_case("16_Decoder_Deconv", {
        "dec_strategy": "deconv"
    })
    
    run_case("17_Decoder_Deep_Hidden", {
        "dec_hidden_layers_num": 2,
        "dec_hidden_ch": 16,
        "static_ch": 2, 
        "head_mode": "diffusion" 
    })
    
    run_case("18_Inference_Mode_Standard", {}, mode='i')
    
    run_case("19_Inference_Mode_Diffusion_Static", {
        "head_mode": "diffusion",
        "static_ch": 4,
        "out_ch": 2
    }, mode='i')

    print("========== Check Finished ==========")

if __name__ == "__main__":
    main()

