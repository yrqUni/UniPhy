import os
import sys
import time
import gc
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

custom_theme = Theme({
    "info": "dim cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "yellow",
    "highlight": "bold magenta"
})
console = Console(theme=custom_theme)

MODEL_CONFIG = {
    "in_channels": 30,
    "out_channels": 30,
    "embed_dim": 768,
    "expand": 4,
    "num_experts": 8,
    "depth": 12,
    "patch_size": 32,
    "img_height": 128,
    "img_width": 256
}

BATCH_SIZE = 1
SEQ_LEN = 4
DT_REF = 6.0

def benchmark_step(model, x, dt, use_amp=False, amp_dtype=torch.bfloat16):
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        pred = model(x, dt)
        loss = pred.mean()
    
    loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) * 1000

def run_test(name, model, x, dt, use_amp=False, compile_mode=None):
    console.print(f"\n[info]Testing {name}...[/]")
    
    if compile_mode:
        try:
            console.print(f"  [dim]Compiling with mode='{compile_mode}'...[/]")
            t0 = time.time()
            model = torch.compile(model, mode=compile_mode)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _ = model(x, dt).mean().backward()
            torch.cuda.synchronize()
            console.print(f"  [success]Compilation done in {time.time()-t0:.2f}s[/]")
        except Exception as e:
            console.print(f"  [error]Compilation Failed![/]\n  {str(e)[:200]}...")
            return None, False

    try:
        for _ in range(3):
            benchmark_step(model, x, dt, use_amp)
    except Exception as e:
        console.print(f"  [error]Runtime Error during warmup![/]\n  {str(e)[:200]}...")
        return None, False

    times = []
    mem_peak = 0
    torch.cuda.reset_peak_memory_stats()
    
    try:
        for _ in range(10):
            t = benchmark_step(model, x, dt, use_amp)
            times.append(t)
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    except Exception as e:
         console.print(f"  [error]Runtime Error during bench![/]\n  {str(e)[:200]}...")
         return None, False

    avg_time = sum(times) / len(times)
    return avg_time, mem_peak

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    if not torch.cuda.is_available():
        console.print("[error]No CUDA device found![/]")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda:0")
    
    console.print("[info]Generating Dummy Data (Resized for Compatibility Check)...[/]")
    x = torch.randn(BATCH_SIZE, SEQ_LEN, MODEL_CONFIG["in_channels"], 
                    MODEL_CONFIG["img_height"], MODEL_CONFIG["img_width"], device=device)
    dt = torch.ones(BATCH_SIZE, SEQ_LEN, device=device) * DT_REF

    table = Table(title="UniPhy Acceleration Benchmark (A800)", header_style="bold magenta")
    table.add_column("Config", style="cyan")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Mem (GB)", justify="right")
    table.add_column("Speedup", style="green", justify="right")
    table.add_column("Status", justify="center")

    console.print("Running Baseline (FP32)...")
    cleanup()
    model = UniPhyModel(**MODEL_CONFIG).to(device)
    base_time, base_mem = run_test("Baseline (FP32)", model, x, dt, use_amp=False)
    del model
    cleanup()

    if base_time:
        table.add_row("FP32 (TF32)", f"{base_time:.1f}", f"{base_mem:.2f}", "1.00x", "[success]OK[/]")
    else:
        table.add_row("FP32 (TF32)", "-", "-", "-", "[error]FAIL[/]")
        return

    console.print("Running AMP (BFloat16)...")
    model = UniPhyModel(**MODEL_CONFIG).to(device)
    amp_time, amp_mem = run_test("AMP (BF16)", model, x, dt, use_amp=True)
    del model
    cleanup()

    if amp_time:
        speedup = base_time / amp_time
        table.add_row("AMP (BF16)", f"{amp_time:.1f}", f"{amp_mem:.2f}", f"{speedup:.2f}x", "[success]OK[/]")
    else:
        table.add_row("AMP (BF16)", "-", "-", "-", "[error]FAIL[/]")

    console.print("Running torch.compile(default)...")
    model_c = UniPhyModel(**MODEL_CONFIG).to(device)
    comp_time, comp_mem = run_test("Compile (Default) + BF16", model_c, x, dt, use_amp=True, compile_mode="default")
    del model_c
    cleanup()
    
    if comp_time:
        speedup = base_time / comp_time
        table.add_row("Compile+BF16", f"{comp_time:.1f}", f"{comp_mem:.2f}", f"{speedup:.2f}x", "[success]OK[/]")
    else:
        table.add_row("Compile+BF16", "-", "-", "-", "[warning]Unsupport[/]")

    console.print("\n")
    console.print(table)
    
    console.print(Panel("[bold]Summary & Recommendation[/]", border_style="blue"))
    if amp_time:
        console.print(f"1. [bold green]BF16 is working![/] Saves {(base_mem - amp_mem):.2f}GB VRAM (on small scale).")
    
    if comp_time and comp_time < amp_time:
        console.print(f"2. [bold green]torch.compile works![/] Further {(amp_time - comp_time):.1f}ms reduction.")
        console.print("   -> In train.yaml: set [bold cyan]use_compile: true[/]")
    elif not comp_time:
        console.print("2. [yellow]torch.compile failed.[/] Your model has dynamic control flow or complex ops.")
        console.print("   -> In train.yaml: set [bold cyan]use_compile: false[/]")
    else:
         console.print("2. [yellow]torch.compile overhead is high on small batch.[/] Verify on real training.")

if __name__ == "__main__":
    main()
    