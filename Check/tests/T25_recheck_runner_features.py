import sys
from pathlib import Path

from Check.utils import run_source_guard, write_result

TEST_ID = "T25"
TARGET = "Check/tests/run_all.py"
T17_TARGET = "Check/tests/T17_numerical_regression.py"


def run():
    status_a, _, detail_a = run_source_guard(
        TEST_ID,
        TARGET,
        {
            "json_out": lambda source: "--json-out" in source,
            "json_write": lambda source: (
                "json.dumps(build_json_report(results, args)" in source
            ),
        },
    )
    status_b, _, detail_b = run_source_guard(
        TEST_ID,
        T17_TARGET,
        {
            "regen_flag": lambda source: "--regenerate" in source,
            "sha256": lambda source: "sha256=" in source,
            "missing_is_fail": lambda source: 'return "FAIL", "-", detail' in source,
        },
    )
    passed = status_a == "PASS" and status_b == "PASS"
    max_err = 0.0 if passed else 1.0
    detail = f"{detail_a} {detail_b}"
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
