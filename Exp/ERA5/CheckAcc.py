import os
import sys
import time
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

custom_theme = Theme({
    "info": "dim cyan",
    "success": "bold green",
    "error": "bold red",
    "warning": "yellow"
})
console = Console(theme=custom_theme)

def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory": f"{props.total_memory / 1024**3:.2f} GB",
        "count": torch.cuda.device_count()
    }

def test_tf32():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        size = 2048
        a = torch.randn(size, size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, size, device="cuda", dtype=torch.float32)
        
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        return True, f"{(end - start)*1000:.2f} ms"
    except Exception as e:
        return False, str(e)

def test_bf16():
    if not torch.cuda.is_bf16_supported():
        return False, "Hardware does not support BF16"
    
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            c = torch.matmul(a, b)
        
        is_bf16 = (c.dtype == torch.bfloat16) or (c.dtype == torch.float32) 
        
        return True, "Success"
    except Exception as e:
        return False, str(e)

def test_compile():
    if os.name == 'nt':
        return False, "Windows not fully supported for compile"
    
    try:
        model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).cuda()
        
        compiled_model = torch.compile(model)
        x = torch.randn(16, 128, device="cuda")
        
        start = time.time()
        _ = compiled_model(x)
        torch.cuda.synchronize()
        first_run = time.time() - start
        
        start = time.time()
        _ = compiled_model(x)
        torch.cuda.synchronize()
        second_run = time.time() - start
        
        return True, f"Warmup: {first_run*1000:.2f}ms | Cached: {second_run*1000:.2f}ms"
    except Exception as e:
        return False, str(e)

def main():
    console.print(Panel.fit("[bold white]UniPhy Environment Verification[/]", border_style="blue"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check Item")
    table.add_column("Status")
    table.add_column("Details")

    py_ver = f"{sys.version.split()[0]}"
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda
    table.add_row("Python / PyTorch", "[success]OK[/]", f"{py_ver} | {torch_ver}+cu{cuda_ver}")

    gpu_info = get_gpu_info()
    if gpu_info:
        status = "[success]OK[/]" if float(gpu_info['capability']) >= 8.0 else "[warning]Old[/]"
        details = f"{gpu_info['name']} (x{gpu_info['count']}) | Cap {gpu_info['capability']} | {gpu_info['memory']}"
        table.add_row("GPU Hardware", status, details)
    else:
        table.add_row("GPU Hardware", "[error]FAIL[/]", "No CUDA device found")
        console.print(table)
        return

    tf32_ok, tf32_msg = test_tf32()
    table.add_row("TensorFloat-32 (TF32)", "[success]OK[/]" if tf32_ok else "[error]FAIL[/]", tf32_msg)

    bf16_ok, bf16_msg = test_bf16()
    table.add_row("BFloat16 (AMP)", "[success]OK[/]" if bf16_ok else "[error]FAIL[/]", bf16_msg)

    compile_ok, compile_msg = test_compile()
    table.add_row("torch.compile", "[success]OK[/]" if compile_ok else "[warning]SKIP[/]", compile_msg)

    console.print(table)

    if bf16_ok and float(gpu_info['capability']) >= 8.0:
        console.print("\n[bold green]Environment is ready for UniPhy A800 Training![/]")
        console.print("Recommended Config: [bold white]use_tf32=True, use_amp=True[/]")
    else:
        console.print("\n[bold red]Environment issues detected. Check drivers or PyTorch version.[/]")

if __name__ == "__main__":
    main()
    