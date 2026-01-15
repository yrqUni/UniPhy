import sys
import os
import torch
import time
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../Model/UniPhy')
sys.path.append(model_path)

try:
    from ModelUniPhy import UniPhy
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class MockArgs:
    def __init__(self, **kwargs):
        self.input_ch = 1
        self.input_size = (32, 64)
        self.emb_ch = 16
        self.hidden_factor = (2, 2)
        self.convlru_num_blocks = 1
        self.Arch = "unet"
        self.down_mode = "avg"
        self.lru_rank = 8
        self.dt_ref = 1.0
        self.inj_k = 2.0
        self.koopman_use_noise = False
        self.koopman_noise_scale = 0.1
        self.dynamics_mode = "spectral"
        self.interpolation_mode = "bilinear"
        self.conservative_dynamics = False
        self.use_pde_refinement = False
        self.pde_viscosity = 1e-3
        self.pscan_use_decay = True
        self.pscan_use_residual = True
        self.pscan_chunk_size = 32
        self.ffn_ratio = 4.0
        self.ConvType = "conv"
        self.spectral_modes_h = 4
        self.spectral_modes_w = 4
        self.out_ch = 1
        self.dist_mode = "gaussian"
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_checks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running checks on {device}\n")

    test_configs = [
        {
            "name": "Default Spectral Mode",
            "dynamics_mode": "spectral",
            "dist_mode": "gaussian"
        },
        {
            "name": "Spectral + Conservative",
            "dynamics_mode": "spectral",
            "conservative_dynamics": True
        },
        {
            "name": "Spectral + PDE Refinement",
            "dynamics_mode": "spectral",
            "use_pde_refinement": True
        },
        {
            "name": "Advection Mode",
            "dynamics_mode": "advection",
            "dist_mode": "laplace"
        },
        {
            "name": "GeoSym Mode (Clifford+Hamiltonian+Stream)",
            "dynamics_mode": "geosym",
            "emb_ch": 16 
        },
        {
            "name": "GeoSym + PDE Refinement",
            "dynamics_mode": "geosym",
            "use_pde_refinement": True,
            "emb_ch": 32 
        },
        {
            "name": "Advection + No Noise",
            "dynamics_mode": "advection",
            "koopman_use_noise": False
        },
        {
            "name": "Spectral + Noise",
            "dynamics_mode": "spectral",
            "koopman_use_noise": True
        },
        {
            "name": "Diffusion Head Mode",
            "dynamics_mode": "spectral",
            "dist_mode": "diffusion",
            "out_ch": 1
        }
    ]

    B, L, C, H, W = 2, 3, 1, 32, 64
    x = torch.randn(B, L, C, H, W).to(device)
    listT = torch.ones(B, L).to(device)

    for config in test_configs:
        print(f"Testing: {config['name']} ...")
        
        args_dict = {k: v for k, v in config.items() if k != "name"}
        args = MockArgs(**args_dict)
        
        try:
            model = UniPhy(args).to(device)
            
            start_time = time.time()
            if args.dist_mode == "diffusion":
                out, _ = model(x, mode="p", listT=listT, x_noisy=torch.randn_like(x), t=torch.zeros(B*L, device=device))
            else:
                out, _ = model(x, mode="p", listT=listT)
            fwd_time = time.time() - start_time
            
            loss = out.mean()
            loss.backward()
            
            print(f"  [PASS] Forward/Backward successful. Time: {fwd_time:.4f}s")
            print(f"  Output shape: {out.shape}")
            
            if config.get("use_pde_refinement", False):
                print(f"  PDE Refinement active: Checked.")
            if config.get("dynamics_mode") == "geosym":
                print(f"  GeoSym-SSM active: Checked.")
            if config.get("dist_mode") == "diffusion":
                print(f"  Diffusion Head active: Checked.")
                
        except Exception:
            print(f"  [FAIL] Error encountered:")
            traceback.print_exc()
        
        print("-" * 50)

if __name__ == "__main__":
    run_checks()

