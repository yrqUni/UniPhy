import os
import sys
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append("/nfs/UniPhy/Model/UniPhy")
sys.path.append("/nfs/UniPhy/Exp/ERA5")

from ModelUniPhy import UniPhyModel
from ERA5 import ERA5_Dataset

def load_config_and_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return None, None

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "cfg" in checkpoint:
        model_cfg = checkpoint["cfg"]["model"]
    else:
        model_cfg = {
            "in_channels": 30, "out_channels": 30, "embed_dim": 512,
            "expand": 4, "depth": 8, "patch_size": (7, 15),
            "img_height": 721, "img_width": 1440, "dt_ref": 6.0,
            "sde_mode": "sde", "init_noise_scale": 0.0001,
            "max_growth_rate": 0.3, "ensemble_size": 4
        }
    
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if "encoder.pos_emb" in state_dict:
        model_cfg["embed_dim"] = state_dict["encoder.pos_emb"].shape[1]

    valid_args = {
        "in_channels", "out_channels", "embed_dim", "expand", "depth",
        "patch_size", "img_height", "img_width", "dt_ref", "sde_mode",
        "init_noise_scale", "ensemble_size", "max_growth_rate", "num_experts"
    }
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in valid_args}

    model = UniPhyModel(**filtered_cfg).to(device)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, model_cfg

def get_test_data():
    try:
        dataset = ERA5_Dataset(
            input_dir="/nfs/ERA5_data/data_norm",
            year_range=[2009, 2009],
            window_size=4,
            sample_k=3,
            sampling_mode="sequential",
            is_train=False
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        batch = next(iter(loader))
        data, _ = batch 
        
        x_ctx = data[:, 0:1].cuda()
        x_target = data[:, 2].cuda()
        
        print("✅ Real ERA5 data loaded successfully.")
        return x_ctx, x_target
    except Exception as e:
        print(f"⚠️ Data loading failed ({e}). Using random tensors.")
        x_ctx = torch.randn(1, 1, 30, 721, 1440).cuda()
        x_target = torch.randn(1, 30, 721, 1440).cuda()
        return x_ctx, x_target

def run_inference_aligned(model, x_ctx, dt_step):
    device = x_ctx.device
    target_dt = 6.0
    
    total_hours = 12.0
    n_iters = int(total_hours / dt_step)
    
    dt_list = [
        torch.tensor(float(dt_step), device=device, dtype=torch.float32)
        for _ in range(n_iters)
    ]

    pred_seq = model.forward_rollout(x_ctx, target_dt, dt_list)
    
    return pred_seq[:, -1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("🧪 Experiment 4: Temporal Consistency (Using forward_rollout)")
    print("=" * 65)

    x_ctx, x_target = get_test_data()

    ckpt_pre_list = sorted(glob.glob("./uniphy/ckpt/*.pt"), key=os.path.getmtime)
    ckpt_align_list = sorted(glob.glob("./uniphy/align_ckpt/*.pt"), key=os.path.getmtime)

    checkpoints = {
        "Pre-trained": ckpt_pre_list[-1] if ckpt_pre_list else None,
        "Fine-tuned": ckpt_align_list[-1] if ckpt_align_list else None
    }

    results = {}
    dt_list = [1, 2, 3, 6]

    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            results[name] = {dt: float('nan') for dt in dt_list}
            continue
            
        print(f"🔄 Evaluating {name} Model...")
        print(f"   Path: {ckpt_path}")
        
        model, _ = load_config_and_model(ckpt_path, device)
        if model is None: continue
            
        model_res = {}
        with torch.no_grad():
            for dt in dt_list:
                try:
                    x_pred = run_inference_aligned(model, x_ctx, dt)
                    
                    mse = torch.mean((x_pred - x_target) ** 2)
                    rmse = torch.sqrt(mse).item()
                    
                    model_res[dt] = rmse
                    print(f"   -> dt={dt}h ({12//dt} steps): RMSE = {rmse:.4f}")
                    
                except Exception as e:
                    print(f"   -> dt={dt}h: Failed ({e})")
                    model_res[dt] = float('nan')
        
        results[name] = model_res
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 65)
    print(f"{'Model Type':<15} | {'dt=1h':<10} | {'dt=2h':<10} | {'dt=3h':<10} | {'dt=6h':<10}")
    print("-" * 65)
    
    for name in ["Pre-trained", "Fine-tuned"]:
        row = f"{name:<15} | "
        res = results.get(name, {})
        for dt in dt_list:
            val = res.get(dt, float('nan'))
            if np.isnan(val):
                row += f"{'N/A':<10} | "
            else:
                row += f"{val:<10.4f} | "
        print(row)
    print("=" * 65)
    print("Method: infer_model.forward_rollout(x_ctx, target_dt, dt_list)")

if __name__ == "__main__":
    main()

