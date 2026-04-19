import argparse
import importlib
import socket
from pathlib import Path

from dt_check.utils import LOG_DIR, format_max_error, write_result

TEST_MODULES = [
    ("T01", "T01_phi1_stability", "T01_phi1_stability"),
    ("T02", "T02_ssm_discretisation", "T02_ssm_discretisation"),
    ("T03", "T03_dt_zero_limit", "T03_dt_zero_limit"),
    ("T04", "T04_variable_dt_pscan", "T04_variable_dt_pscan"),
    (
        "T05",
        "T05_hprev_injection_equivalence",
        "T05_hprev_injection_equivalence",
    ),
    ("T06", "T06_flux_prev_compensation", "T06_flux_prev_compensation"),
    ("T07", "T07_cumprod_decay_purity", "T07_cumprod_decay_purity"),
    ("T08", "T08_rollout_dt_alignment", "T08_rollout_dt_alignment"),
    ("T09", "T09_forward_vs_step_single", "T09_forward_vs_step_single"),
    (
        "T10",
        "T10_forward_vs_rollout_multistep",
        "T10_forward_vs_rollout_multistep",
    ),
    (
        "T11",
        "T11_basis_encode_decode_identity",
        "T11_basis_encode_decode_identity",
    ),
    ("T12", "T12_basis_biorthogonality", "T12_basis_biorthogonality"),
    ("T13", "T13_sde_scale_physics", "T13_sde_scale_physics"),
    ("T14", "T14_gradient_flow", "T14_gradient_flow"),
    ("T15", "T15_dt_zero_mask", "T15_dt_zero_mask"),
    ("T16", "T16_negative_dt_rejection", "T16_negative_dt_rejection"),
    ("T17", "T17_numerical_regression", "T17_numerical_regression"),
]
TEST_LOOKUP = {}
for test_id, module_name, result_id in TEST_MODULES:
    TEST_LOOKUP[test_id] = (module_name, result_id)
    TEST_LOOKUP[module_name] = (module_name, result_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--log-dir", default=LOG_DIR)
    return parser.parse_args()


def resolve_tests(requested):
    if not requested:
        return TEST_MODULES
    resolved = []
    for name in requested:
        if name not in TEST_LOOKUP:
            raise ValueError(f"unknown test: {name}")
        module_name, result_id = TEST_LOOKUP[name]
        resolved.append((name, module_name, result_id))
    return resolved


def main():
    args = parse_args()
    results = []
    selected = resolve_tests(args.tests)
    for _, module_name, result_id in selected:
        module = importlib.import_module(f"dt_check.tests.{module_name}")
        try:
            status, max_error, detail = module.run()
        except Exception as exc:
            status = "FAIL"
            max_error = float("nan")
            detail = f"{type(exc).__name__}: {exc}"
        write_result(result_id, status, max_error, detail, log_dir=args.log_dir)
        results.append((status, result_id, max_error))
    pass_count = sum(status == "PASS" for status, _, _ in results)
    fail_count = sum(status == "FAIL" for status, _, _ in results)
    skip_count = sum(status == "SKIP" for status, _, _ in results)
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    summary_path = Path(args.log_dir) / f"node_{hostname}_summary.txt"
    summary_lines = [
        f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  TOTAL: 17"
    ]
    summary_lines.extend(
        f"{status} {test_id} max_error={format_max_error(max_error)}"
        for status, test_id, max_error in results
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
