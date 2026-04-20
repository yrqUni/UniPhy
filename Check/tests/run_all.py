import argparse
import importlib
import socket
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import LOG_DIR, format_max_error, write_result
else:
    from ..utils import LOG_DIR, format_max_error, write_result

TEST_MODULES = [
    ("T01", "T01_phi1_stability", "T"),
    ("T02", "T02_ssm_discretisation", "T"),
    ("T03", "T03_dt_zero_limit", "T"),
    ("T04", "T04_variable_dt_pscan", "T"),
    ("T05", "T05_hprev_injection_equivalence", "T"),
    ("T06", "T06_flux_prev_compensation", "T"),
    ("T07", "T07_cumprod_decay_purity", "T"),
    ("T08", "T08_rollout_dt_alignment", "T"),
    ("T09", "T09_forward_vs_step_single", "T"),
    ("T10", "T10_forward_vs_rollout_multistep", "T"),
    ("T11", "T11_basis_encode_decode_identity", "T"),
    ("T12", "T12_basis_biorthogonality", "T"),
    ("T13", "T13_sde_scale_physics", "T"),
    ("T14", "T14_gradient_flow", "T"),
    ("T15", "T15_dt_zero_mask", "T"),
    ("T16", "T16_negative_dt_rejection", "T"),
    ("T17", "T17_numerical_regression", "T"),
    ("S01", "S01_parallel_serial_consistency", "S"),
    ("S02", "S02_timestep_semantics", "S"),
    ("S03", "S03_parameter_consistency", "S"),
    ("S04", "S04_architecture_verification", "S"),
    ("S05", "S05_pscan_correctness", "S"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--log-dir", default=LOG_DIR)
    return parser.parse_args()


def resolve_tests(requested):
    if not requested:
        return TEST_MODULES
    selected = []
    for name in requested:
        if name in {"T", "S"}:
            matches = [entry for entry in TEST_MODULES if entry[0].startswith(name)]
        else:
            matches = [
                entry for entry in TEST_MODULES if entry[0] == name or entry[1] == name
            ]
        if not matches:
            raise ValueError(f"unknown test: {name}")
        for entry in matches:
            if entry not in selected:
                selected.append(entry)
    return selected


def main():
    args = parse_args()
    results = []
    selected = resolve_tests(args.tests)
    for test_id, module_name, series in selected:
        module = importlib.import_module(f"Check.tests.{module_name}")
        try:
            if series == "S":
                status, pass_count, total = module.run()
                max_error = "-"
                detail = f"pass_count={pass_count} total={total}"
            else:
                status, max_error, detail = module.run()
            write_result(test_id, status, max_error, detail, log_dir=args.log_dir)
        except Exception as exc:
            status = "FAIL"
            max_error = "-"
            detail = f"{type(exc).__name__}: {exc}"
            write_result(test_id, status, max_error, detail, log_dir=args.log_dir)
        results.append((status, test_id, max_error))
    pass_count = sum(status == "PASS" for status, _, _ in results)
    fail_count = sum(status == "FAIL" for status, _, _ in results)
    skip_count = sum(status == "SKIP" for status, _, _ in results)
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    summary_path = Path(args.log_dir) / f"node_{hostname}_summary.txt"
    summary_lines = [
        (
            f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  "
            f"TOTAL: {len(results)}"
        )
    ]
    summary_lines.extend(
        f"{status} {test_id} max_error={format_max_error(max_error)}"
        for status, test_id, max_error in results
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
