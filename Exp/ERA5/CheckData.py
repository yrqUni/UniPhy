import gc
import sys
import torch
from rich.console import Console
from rich.table import Table

sys.path.append("/nfs/UniPhy/Model/UniPhy")
from ModelUniPhy import UniPhyModel

console = Console()


def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


def check_config(depth, dim, expand, num_experts, h, w, patch_size=32):
    model = None
    try:
        torch.cuda.empty_cache()
        gc.collect()

        model = UniPhyModel(
            in_channels=30,
            out_channels=30,
            embed_dim=dim,
            expand=expand,
            num_experts=num_experts,
            depth=depth,
            patch_size=patch_size,
            img_height=h,
            img_width=w,
            dt_ref=6.0,
            sde_mode="det",
            init_noise_scale=0.01,
            max_growth_rate=0.3,
        ).cuda()

        params = get_model_size(model)

        x = torch.randn(1, 4, 30, h, w).cuda()
        dt = torch.ones(4).cuda() * 6.0

        torch.cuda.reset_peak_memory_stats()

        out = model(x, dt)

        if out.is_complex():
            loss = out.abs().mean()
        else:
            loss = out.mean()

        loss.backward()

        mem = torch.cuda.max_memory_allocated() / 1024 ** 3

        del model, x, dt, out, loss
        torch.cuda.empty_cache()
        gc.collect()

        return params, mem, True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()
            return 0, 0, False
        else:
            raise e


def main():
    H, W = 721, 1440

    configs = [
        (4, 256, 4, 4),
        (4, 384, 4, 4),
        (6, 384, 4, 4),
        (6, 512, 4, 4),
        (8, 384, 4, 4),
        (8, 512, 4, 8),
        (8, 768, 4, 8),
    ]

    table = Table(title="UniPhy Config Check", header_style="bold magenta")
    table.add_column("Depth", justify="center")
    table.add_column("Dim", justify="center")
    table.add_column("Expand", justify="center")
    table.add_column("Experts", justify="center")
    table.add_column("Params (M)", justify="right")
    table.add_column("Mem (GB)", justify="right")
    table.add_column("Status", justify="center")

    for d, dim, exp, num_exp in configs:
        p, m, success = check_config(d, dim, exp, num_exp, H, W)
        status = "[green]OK[/green]" if success else "[red]OOM[/red]"
        params_str = f"{p/1e6:.2f}" if success else "-"
        mem_str = f"{m:.2f}" if success else "-"
        table.add_row(
            str(d), str(dim), str(exp), str(num_exp), params_str, mem_str, status
        )

    console.print(table)

    console.print("\n[bold cyan]Recommended Configs by GPU Memory:[/bold cyan]")
    console.print("  24GB (RTX 3090/4090): depth=4, dim=256, experts=4")
    console.print("  40GB (A100-40G): depth=6, dim=384, experts=4")
    console.print("  80GB (A100-80G): depth=8, dim=512, experts=8")


if __name__ == "__main__":
    main()
    