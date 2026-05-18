import argparse
import csv
from pathlib import Path

from Exp.Ablation.protocol import (
    DEFAULT_METRICS,
    DEFAULT_PRIMARY_LEADS,
    VARIANT_ORDER,
    get_variant_spec,
    load_results,
    summarize_results,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate UniPhy ablation results across variants and seeds."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-variant evaluation JSON files.",
    )
    parser.add_argument(
        "--lead-times",
        default=",".join(str(x) for x in DEFAULT_PRIMARY_LEADS),
        help="Comma-separated lead times to include in the table.",
    )
    parser.add_argument(
        "--metric",
        default="rmse",
        choices=DEFAULT_METRICS,
        help="Primary metric used for best-value highlighting.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-tex", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _parse_lead_times(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _fmt(value, std=None):
    if value is None:
        return "-"
    if std is None:
        return f"{value:.4f}"
    return f"{value:.4f} +/- {std:.4f}"


def _metric_cols(metric, lead_times):
    return [f"{metric}@{lead}h" for lead in lead_times]


def _best_values(rows, metric, lead_times):
    best = {}
    for col in _metric_cols(metric, lead_times):
        vals = [row[col] for row in rows if row.get(col) is not None]
        if not vals:
            best[col] = None
        elif metric == "acc":
            best[col] = max(vals)
        else:
            best[col] = min(vals)
    return best


def print_table(rows, lead_times, metric):
    cols = _metric_cols(metric, lead_times)
    best = _best_values(rows, metric, lead_times)
    header = f"{'Variant':<28} {'n':>3}" + "".join(f"  {col:>20}" for col in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['label']:<28} {row['n']:>3}"
        for col in cols:
            value = row.get(col)
            std = row.get(f"{col}_std")
            cell = _fmt(value, std)
            if value is not None and best.get(col) is not None:
                if abs(value - best[col]) < 1e-12:
                    cell = "*" + cell
            line += f"  {cell:>20}"
        print(line)


def write_csv(rows, lead_times, path):
    metric_fields = []
    for metric in DEFAULT_METRICS:
        for lead in lead_times:
            key = f"{metric}@{lead}h"
            metric_fields.extend(
                [
                    key,
                    f"{key}_std",
                    f"{key}_ci95",
                    f"{key}_delta_vs_baseline",
                ]
            )
    cols = ["variant", "label", "group", "n"] + metric_fields
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV -> {path}")


def write_latex(rows, lead_times, path, metric):
    cols = _metric_cols(metric, lead_times)
    best = _best_values(rows, metric, lead_times)
    lines = [
        r"\begin{tabular}{ll" + "r" * len(cols) + "}",
        r"\toprule",
        "Group & Variant & " + " & ".join(f"{lead}h" for lead in lead_times) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["group"], row["label"]]
        for col in cols:
            value = row.get(col)
            std = row.get(f"{col}_std")
            if value is None:
                cell = "-"
            else:
                cell = f"{value:.4f}"
                if std is not None:
                    cell += f" $\\pm$ {std:.4f}"
                if best.get(col) is not None and abs(value - best[col]) < 1e-12:
                    cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LaTeX -> {path}")


def main():
    args = parse_args()
    lead_times = _parse_lead_times(args.lead_times)
    results = load_results(args.results_dir)
    rows = summarize_results(results, lead_times)

    if not rows:
        print(f"No valid evaluation JSON files found in {args.results_dir}")
        return

    print(f"\nAblation comparison: {args.metric.upper()} @ {lead_times}h\n")
    print_table(rows, lead_times, args.metric)

    if args.output_csv:
        write_csv(rows, lead_times, args.output_csv)
    if args.output_tex:
        write_latex(rows, lead_times, args.output_tex, args.metric)
    if args.output_json:
        write_json(
            args.output_json,
            {
                "variant_order": list(VARIANT_ORDER),
                "variants": {
                    name: get_variant_spec(name).to_dict() for name in VARIANT_ORDER
                },
                "lead_times_hours": lead_times,
                "rows": rows,
            },
        )
        print(f"JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
