import argparse
from pathlib import Path

from Exp.Ablation.protocol import (
    DEFAULT_LEAD_TIMES,
    DEFAULT_SEEDS,
    protocol_dict,
    variant_names,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible UniPhy ablation execution plan."
    )
    parser.add_argument("--output-dir", default="Exp/Ablation/results")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'. Includes baseline by default.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds for paired repeated runs.",
    )
    parser.add_argument(
        "--lead-times",
        default=",".join(str(lead) for lead in DEFAULT_LEAD_TIMES),
        help="Comma-separated evaluation lead times in hours.",
    )
    parser.add_argument("--data-input-dir", default="/data/ERA5")
    parser.add_argument("--train-year-range", default="2000,2016")
    parser.add_argument("--eval-year-range", default="2017,2018")
    parser.add_argument("--climatology-year-range", default="2000,2016")
    parser.add_argument("--epochs", default="<N>")
    parser.add_argument("--gpus", default="<gpus>")
    return parser.parse_args()


def _parse_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_variants(text):
    if text.strip().lower() == "all":
        return variant_names(include_baseline=True)
    names = [x.strip() for x in text.split(",") if x.strip()]
    known = set(variant_names(include_baseline=True))
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return names


def build_plan(args):
    seeds = _parse_ints(args.seeds)
    leads = _parse_ints(args.lead_times)
    variants = _parse_variants(args.variants)
    out_root = Path(args.output_dir)
    jobs = []
    for variant in variants:
        for seed in seeds:
            run_dir = out_root / variant / f"seed_{seed}"
            train_cmd = [
                "torchrun",
                f"--nproc_per_node={args.gpus}",
                "-m",
                "Exp.Ablation.runner",
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--data-input-dir",
                args.data_input_dir,
                "--train-year-range",
                args.train_year_range,
                "--epochs",
                str(args.epochs),
                "--ckpt-dir",
                str(run_dir),
            ]
            eval_cmd = [
                "python",
                "-m",
                "Exp.Ablation.eval",
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--checkpoint",
                str(run_dir / "ckpt_final.pt"),
                "--data-input-dir",
                args.data_input_dir,
                "--climatology-dir",
                args.data_input_dir,
                "--climatology-year-range",
                args.climatology_year_range,
                "--eval-year-range",
                args.eval_year_range,
                "--lead-times",
                ",".join(str(lead) for lead in leads),
                "--output-json",
                str(run_dir / "eval.json"),
            ]
            jobs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "train_command": train_cmd,
                    "eval_command": eval_cmd,
                }
            )
    return {
        "protocol": protocol_dict(seeds=seeds, lead_times=leads),
        "jobs": jobs,
        "compare_command": [
            "python",
            "-m",
            "Exp.Ablation.compare",
            "--results-dir",
            str(out_root),
            "--lead-times",
            "24,120,240",
            "--metric",
            "rmse",
            "--output-csv",
            str(out_root / "summary.csv"),
            "--output-tex",
            str(out_root / "summary.tex"),
            "--output-json",
            str(out_root / "summary.json"),
        ],
    }


def main():
    args = parse_args()
    plan = build_plan(args)
    out_path = Path(args.output_dir) / "protocol.json"
    write_json(out_path, plan)
    print(f"Wrote ablation protocol -> {out_path}")
    print(f"Planned jobs: {len(plan['jobs'])}")


if __name__ == "__main__":
    main()
