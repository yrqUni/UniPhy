import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from Exp.ERA5.align import align_step
from Exp.ERA5.train import build_lat_weights as build_train_lat_weights
from Exp.ERA5.train import compute_crps as train_compute_crps
from Model.UniPhy.ModelUniPhy import UniPhyModel

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CFG_PATH = ROOT / "Exp/ERA5/train.yaml"
ALIGN_CFG_PATH = ROOT / "Exp/ERA5/align.yaml"


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Recorder:
    def __init__(self, category_name):
        self.category_name = category_name
        self.total = 0
        self.failed = 0
        self.failures = []

    def check(self, name, passed, detail):
        self.total += 1
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            self.failed += 1
            self.failures.append(name)

    def finalize(self):
        passed = self.total - self.failed
        print(f"CATEGORY {self.category_name}: {passed}/{self.total} passed")
        if self.failed:
            print(
                f"CATEGORY {self.category_name} FAILURES: "
                f"{', '.join(self.failures)}"
            )
            raise SystemExit(1)
        print(f"CATEGORY {self.category_name}: ALL PASS")
        raise SystemExit(0)


def run_check(recorder, name, fn):
    try:
        detail = fn()
        recorder.check(name, True, str(detail))
    except Exception as exc:
        recorder.check(name, False, f"{type(exc).__name__}: {exc}")


def load_yaml_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_validation_cfg(cfg, patch_grid=4):
    cfg = copy.deepcopy(cfg)
    patch_h, patch_w = tuple(cfg["model"]["patch_size"])
    cfg["model"]["img_height"] = patch_h * patch_grid
    cfg["model"]["img_width"] = patch_w * patch_grid
    return cfg


def build_train_cfg(patch_grid=4):
    return make_validation_cfg(load_yaml_file(TRAIN_CFG_PATH), patch_grid=patch_grid)


def build_align_cfg(patch_grid=4):
    return make_validation_cfg(load_yaml_file(ALIGN_CFG_PATH), patch_grid=patch_grid)


def build_model(model_cfg, device):
    model = UniPhyModel(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        embed_dim=model_cfg["embed_dim"],
        expand=model_cfg["expand"],
        depth=model_cfg["depth"],
        patch_size=tuple(model_cfg["patch_size"]),
        img_height=model_cfg["img_height"],
        img_width=model_cfg["img_width"],
        dt_ref=model_cfg["dt_ref"],
        init_noise_scale=model_cfg["init_noise_scale"],
    )
    return model.to(device)


def make_synthetic_batch(
    model_cfg,
    device,
    batch_size=1,
    time_steps=4,
    scale=1.0,
):
    height = int(model_cfg["img_height"])
    width = int(model_cfg["img_width"])
    channels = int(model_cfg["in_channels"])
    dt_ref = float(model_cfg["dt_ref"])
    shape = (batch_size, time_steps, channels, height, width)
    data = torch.randn(shape, device=device, dtype=torch.float32) * float(scale)
    base_dt = torch.tensor(
        [dt_ref, dt_ref / 2.0, dt_ref * 1.5, dt_ref * 0.75],
        device=device,
        dtype=torch.float32,
    )
    repeats = (time_steps + base_dt.numel() - 1) // base_dt.numel()
    dt = base_dt.repeat(repeats)[:time_steps]
    dt = dt.unsqueeze(0).repeat(batch_size, 1).contiguous()
    return data, dt


def compute_stage1_terms(model, cfg, data, dt):
    device = data.device
    x_input = data[:, :-1]
    x_target = data[:, 1:]
    dt_input = dt[:, 1:]
    lat_weights = build_train_lat_weights(
        x_target.shape[-2],
        x_target.shape[-1],
        device,
    )
    ensemble_preds = []
    ensemble_size = int(cfg["model"]["ensemble_size"])
    for _ in range(ensemble_size):
        noise = model.sample_noise(x_input)
        out = model(x_input, dt_input, z=noise)
        if out.is_complex():
            out = out.real
        ensemble_preds.append(out)
    ensemble_stack = torch.stack(ensemble_preds, dim=0)
    out_mean = ensemble_stack.mean(dim=0)
    l1 = ((out_mean - x_target).abs() * lat_weights).mean()
    mse = ((out_mean - x_target) ** 2 * lat_weights).mean()
    crps = train_compute_crps(ensemble_stack, x_target)
    loss = l1 + crps
    return {
        "ensemble_stack": ensemble_stack,
        "l1": l1,
        "mse": mse,
        "crps": crps,
        "loss": loss,
    }


def compute_alignment_metrics(model, cfg, data, dt, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    return align_step(
        model,
        (data, dt),
        optimizer,
        cfg,
        grad_accum_steps=2,
        batch_idx=0,
        epoch=0,
        total_epochs=1,
    )


def build_train_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )


def make_fast_align_cfg(
    cfg,
    condition_steps=2,
    max_target_steps=1,
    sub_steps=(1,),
    max_rollout_steps=1,
    chunk_size=1,
):
    cfg = copy.deepcopy(cfg)
    cfg["alignment"]["condition_steps"] = int(condition_steps)
    cfg["alignment"]["max_target_steps"] = int(max_target_steps)
    cfg["alignment"]["sub_steps"] = [int(step) for step in sub_steps]
    cfg["alignment"]["max_rollout_steps"] = int(max_rollout_steps)
    cfg["alignment"]["chunk_size"] = int(chunk_size)
    return cfg


def list_project_modules(root):
    root = Path(root)
    modules = []
    for path in sorted(root.glob("**/*.py")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(part == "__pycache__" for part in rel.parts):
            continue
        if rel.name == "__init__.py":
            module_parts = rel.parts[:-1]
        else:
            module_parts = rel.with_suffix("").parts
        if not module_parts:
            continue
        modules.append(".".join(module_parts))
    return modules


def gaussian_crps_closed_form(mu, sigma, target):
    z = (target - mu) / sigma
    sqrt_pi = math.sqrt(math.pi)
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt_pi)


def gaussian_quantile_ensemble(mu, sigma, members, device):
    u = (
        torch.arange(members, device=device, dtype=torch.float64) + 0.5
    ) / members
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    samples = mu + sigma * z
    return samples.reshape(members, 1, 1)


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finite_number(value):
    return math.isfinite(float(value))
