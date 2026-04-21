import sys
from pathlib import Path

from Check.utils import run_source_guard, write_result

TEST_ID = "T21"
TARGET = "Exp/ERA5/align.py"


def run():
    return run_source_guard(
        TEST_ID,
        TARGET,
        {
            "pred_mean_removed": lambda source: (
                "pred_mean = (pred_seq + other_sum) / ensemble_size" not in source
            ),
            "delete_removed": lambda source: "del pred_mean" not in source,
        },
    )


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
