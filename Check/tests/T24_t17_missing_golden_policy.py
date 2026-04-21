import sys
from pathlib import Path

from Check.utils import run_source_guard, write_result

TEST_ID = "T24"
TARGET = "Check/tests/T17_numerical_regression.py"


def run():
    return run_source_guard(
        TEST_ID,
        TARGET,
        {
            "autosave": lambda source: (
                'torch.save({"fwd": out_fwd_r.cpu(), "roll": out_roll_r.cpu()}, '
                'GOLDEN_PATH)'
                not in source
            ),
            "no_pass_on_missing": lambda source: (
                'return "PASS", 0.0, "golden values saved"' not in source
            ),
            "missing_is_fail": lambda source: 'return "FAIL", "-", detail' in source,
        },
    )


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
