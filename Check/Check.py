import sys
import os
import torch
import torch.nn as nn
import traceback

# Add paths to ensure imports work correctly
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
        # Default Basic Arguments
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
        self.dec_strategy = "pxsf"
        self.dec_hidden_ch = 0
        self.dec_hidden_layers_num = 0
        self.head_mode = "gaussian"
        self.diffusion_steps = 5
        self.use_checkpointing = False
        self.ConvType = "conv"
        self.Arch = "unet"
        
        # Advanced Physics & SSM Arguments (New Features)
        self.use_spectral_mixing = False
        self.use_anisotropic_diffusion = False
        self.use_advection = False          # Tests AdvectionBlock & Lagrangian logic
        self.use_graph_interaction = False
        self.use_adaptive_ssm = False       # Tests input/output gating
        self.use_neural_operator = False
        self.learnable_init_state = False
        self.use_wavelet_ssm = False
        self.use_cross_var_attn = False
        
        # Priors
        self.use_freq_prior = False
        self.freq_rank = 4
        self.freq_gain_init = 0.0
        self.freq_mode = "linear"
        self.use_sh_prior = False
        self.sh_Lmax = 4
        self.sh_rank = 4
        self.sh_gain_init = 0.0

        # Override defaults with provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_case(case_name, args_dict, input_shape=(2, 4, 2, 32, 32), mode='p'):
    print(f"[{case_name}] Setting up...")
    args = MockArgs(**args_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = ConvLRU(args).to(device)
        # Verify specific components exist based on args
        if args.use_advection:
            # Check if AdvectionBlock is initialized in layers
            has_adv = hasattr(model.convlru_model.down_blocks[0].lru_layer, 'advection')
            if not has_adv:
                print(f"  [Warning] use_advection=True but 'advection' module not found in layer.")
        
        if args.use_anisotropic_diffusion:
             # Just ensures the parameter generator input dim handles this
             pass

    except Exception as e:
        print(f"[{case_name}] Initialization Failed: {e}")
        traceback.print_exc()
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
                
            # Basic sanity check on output shape vs input
            # If gaussian head, output channel is usually 2*C (mean, std) or defined by out_ch
            expected_out_ch = args.out_ch * 2 if args.head_mode == "gaussian" else args.out_ch
            if not isinstance(out, tuple) and out.shape[2] != expected_out_ch:
                 print(f"  [Warning] Expected output channels {expected_out_ch}, got {out.shape[2]}")

        elif mode == 'i':
            out_gen_num = 3
            listT_future = torch.ones(B, out_gen_num - 1).to(device)
            out = model(x, mode='i', out_gen_num=out_gen_num, listT=listT, listT_future=listT_future, static_feats=static_feats, timestep=timestep)
            print(f"[{case_name}] Inference Success. Output shape: {out.shape}")
            
    except Exception as e:
        print(f"[{case_name}] Forward Execution Failed: {e}")
        traceback.print_exc()
    
    del model, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print("========== Start Checking ConvLRU (Lagrangian & Dynamic SSM) ==========")
    
    # 1. Baseline
    run_case("01_Baseline_UNet", {})
    
    # 2. Physics-Informed Components (The new stuff)
    run_case("02_Lagrangian_Advection", {
        "use_advection": True,  # Trigger AdvectionBlock & LatentFlowPredictor
        "emb_ch": 32 
    })
    
    run_case("03_Anisotropic_Diffusion", {
        "use_anisotropic_diffusion": True, # Trigger Gradient calc in _apply_forcing
        "emb_ch": 32
    })
    
    run_case("04_Spectral_Viscosity_and_Mixing", {
        "use_spectral_mixing": True, # Trigger SpectralInteraction
        "use_freq_prior": True       # Trigger SpectralConv2d
    })
    
    # 3. Dynamic SSM (Mamba-style Gating & LoRA)
    run_case("05_Dynamic_SSM_Gating", {
        "use_adaptive_ssm": True, # Trigger Input/Output gating logic
        "lru_rank": 8
    })
    
    # 4. Complex Interactions
    run_case("06_Graph_and_CrossVar", {
        "use_graph_interaction": True,
        "use_cross_var_attn": True
    })
    
    run_case("07_Wavelet_SSM", {
        "use_wavelet_ssm": True
    })

    # 5. Architecture Variations
    run_case("08_BiFPN_Arch", {
        "Arch": "bifpn"
    })
    
    run_case("09_No_UNet", {
        "Arch": "no_unet",
        "convlru_num_blocks": 2
    })

    # 6. Static Features & Initialization
    run_case("10_Static_Init", {
        "static_ch": 4,
        "learnable_init_state": True
    })

    # 7. Decoder & Head Variations
    run_case("11_Diffusion_Head", {
        "head_mode": "diffusion",
        "out_ch": 2
    })
    
    run_case("12_Token_Head_VQ", {
        "head_mode": "token",
        "out_ch": 2
    })

    # 8. Inference Mode
    run_case("13_Inference_Loop", {
        "use_advection": True # Test advection in inference loop
    }, mode='i')

    # 9. Full Kitchen Sink (Extreme Stress Test)
    run_case("14_All_Features_Combined", {
        "use_advection": True,
        "use_anisotropic_diffusion": True,
        "use_spectral_mixing": True,
        "use_adaptive_ssm": True,
        "use_graph_interaction": True,
        "use_wavelet_ssm": True,
        "use_sh_prior": True,
        "static_ch": 2,
        "learnable_init_state": True,
        "head_mode": "gaussian"
    })

    print("========== Check Finished ==========")

if __name__ == "__main__":
    main()

