import torch
import torch.nn as nn
import pandas as pd
import sys
import os
from collections import OrderedDict

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

ARGS = {
    "H": 721,
    "W": 1440,
    "C": 30,
    "dim": 512,
    "patch_size": 16,
    "num_layers": 4,
    "expansion": 2,
    "bs": 1,
    "T": 4,
    "device": "cuda:0"
}

class MemoryProfiler:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.report = []
        self.param_mem = 0
        
        torch.cuda.reset_peak_memory_stats()
        self.base_mem = torch.cuda.memory_allocated() / 1024**2

    def _get_gpu_mem(self):
        return torch.cuda.memory_allocated() / 1024**2

    def _hook_pre(self, module, input):
        setattr(module, '_mem_prev', self._get_gpu_mem())

    def _hook_post(self, module, input, output):
        mem_prev = getattr(module, '_mem_prev', 0)
        mem_curr = self._get_gpu_mem()
        delta = mem_curr - mem_prev
        
        out_shape = "None"
        if isinstance(output, torch.Tensor):
            out_shape = list(output.shape)
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                out_shape = list(output[0].shape)

        self.report.append({
            "Layer": str(module.__class__.__name__),
            "Name": getattr(module, '_name', 'Unknown'),
            "Input Mem (MB)": round(mem_prev, 2),
            "Post Mem (MB)": round(mem_curr, 2),
            "Delta (Activations) (MB)": round(delta, 2),
            "Output Shape": str(out_shape)
        })

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0 or isinstance(module, (UniPhyModel, nn.Sequential)): 
                module._name = name
                self.hooks.append(module.register_forward_pre_hook(self._hook_pre))
                self.hooks.append(module.register_forward_hook(self._hook_post))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def print_summary(self):
        df = pd.DataFrame(self.report)
        print("\n" + "="*80)
        print(f"GPU Memory Profile Report (Top 20 Consumers)")
        print("="*80)
        
        df_sorted = df.sort_values(by="Delta (Activations) (MB)", ascending=False)
        
        print(df_sorted.head(20).to_string(index=False))
        
        print("-" * 80)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Base Model Weights: {self.base_mem:.2f} MB")
        print(f"Peak Memory During Forward: {peak:.2f} MB")
        print(f"Total Increase: {peak - self.base_mem:.2f} MB")
        print("="*80)

def main():
    torch.cuda.empty_cache()
    device = torch.device(ARGS["device"])

    print(f"Initializing Model with H={ARGS['H']}, W={ARGS['W']}, P={ARGS['patch_size']}...")
    
    model = UniPhyModel(
        input_shape=(ARGS['H'], ARGS['W']),
        in_channels=ARGS['C'],
        dim=ARGS['dim'],
        patch_size=ARGS['patch_size'],
        num_layers=ARGS['num_layers'],
        para_pool_expansion=ARGS['expansion'],
        conserve_energy=True
    ).to(device)
    
    model.eval()

    profiler = MemoryProfiler(model)
    profiler.register_hooks()

    print("Running Forward Pass...")
    x = torch.randn(ARGS['bs'], ARGS['T'], ARGS['C'], ARGS['H'], ARGS['W'], device=device)
    dt = torch.ones(ARGS['bs'], ARGS['T'], device=device) 

    with torch.no_grad():
        try:
            _ = model(x, dt)
        except RuntimeError as e:
            print(f"OOM during profiling! Last captured stats will be printed.\nError: {e}")
    
    profiler.remove_hooks()
    profiler.print_summary()

if __name__ == "__main__":
    main()

