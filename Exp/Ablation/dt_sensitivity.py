import argparse
import copy
from pathlib import Path

from Exp.Ablation.eval import evaluate
from Exp.Ablation.protocol import write_json


DEFAULT_GRIDS = {
    "regular_6h": "6,12,18,24",
    "regular_12h": "12,24,36,48",
    "irregular_short": "3,9,21,33",
    "irregular_medium": "6,18,42,78,120",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ablation sensitivity to inference dt grids.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-input-dir", required=True)
    parser.add_argument("--climatology-dir", default=None)
    parser.add_argument("--climatology-year-range", default=None)
    parser.add_argument("--eval-year-range", default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grids", default=None)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _parse_grids(text):
    if not text:
        return dict(DEFAULT_GRIDS)
    grids = {}
    for item in text.split(";"):
        if not item.strip():
            continue
        name, values = item.split("=", 1)
        grids[name.strip()] = values.strip()
    return grids


def main():
    args = parse_args()
    grids = _parse_grids(args.grids)
    results = {}
    out_path = Path(args.output_json)
    for name, lead_times in grids.items():
        eval_args = copy.copy(args)
        eval_args.lead_times = lead_times
        eval_args.output_json = str(out_path.with_name(f"{out_path.stem}_{name}.json"))
        eval_args.log_every = max(1, int(args.log_every))
        result = evaluate(eval_args)
        results[name] = result
    write_json(
        args.output_json,
        {
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "eval_year_range": args.eval_year_range,
            "max_samples": args.max_samples,
            "grids": grids,
            "results": results,
        },
    )
    print(f"saved -> {args.output_json}")


if __name__ == "__main__":
    main()
