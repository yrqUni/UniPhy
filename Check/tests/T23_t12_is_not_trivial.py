import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import run_source_guard, write_result
else:
    from ..utils import run_source_guard, write_result

TEST_ID = "T23"
TARGET = "Check/tests/T12_basis_biorthogonality.py"


def run():
    return run_source_guard(
        TEST_ID,
        TARGET,
        {
            "constant_pass": lambda source: (
                'return "PASS", residual_100, detail' not in source
            ),
            "broken_case": lambda source: "residual_broken" in source,
            "status_gate": lambda source: (
                'status = "PASS" if passed else "FAIL"' in source
            ),
        },
    )


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
