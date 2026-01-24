import sys
import torch
from rich.console import Console
from rich.table import Table

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

console = Console()


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def check_config(depth, dim, expand, num_experts, h, w):
    try:
        model = UniPhyModel(
            in_channels=30,
            out_channels=30,
            embed_dim=dim,
            expand=expand,
            num_experts=num_experts,
            depth=depth,
            patch_size=32,
            img_height=h,
            img_width=w,
        ).cuda()
        params = get_model_size(model)
        x = torch.randn(1, 4, 30, h, w).cuda()
        dt = torch.ones(1, 4).cuda()
        torch.cuda.reset_peak_memory_stats()
        out = model(x, dt)
        loss = out.mean()
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        del model, x, dt, out, loss
        torch.cuda.empty_cache()
        return params, mem, True
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return 0, 0, False
        else:
            raise e


def main():
    H, W = 721, 1440
    configs = [
        (4, 384, 4, 4),
        (6, 480, 4, 4),
        (6, 512, 4, 4),
        (8, 512, 4, 4),
        (8, 768, 4, 4),
        (12, 768, 4, 8),
    ]
    table = Table(title="Max Params Check (A100-80G Est.)", header_style="bold magenta")
    table.add_column("Depth")
    table.add_column("Dim")
    table.add_column("Experts")
    table.add_column("Params (M)")
    table.add_column("Mem (GB)")
    table.add_column("Status")
    for d, dim, exp, num_exp in configs:
        p, m, success = check_config(d, dim, exp, num_exp, H, W)
        status = "[green]OK[/green]" if success else "[red]OOM[/red]"
        table.add_row(
            str(d), str(dim), str(num_exp), f"{p/1e6:.2f}", f"{m:.2f}", status
        )
    console.print(table)


if __name__ == "__main__":
    main()
    