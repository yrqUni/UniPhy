import sys
from pathlib import Path

from Check.utils import run_source_guard, write_result

TEST_ID = "T22"
TARGET = "Model/UniPhy/PScan.py"


def run():
    return run_source_guard(
        TEST_ID,
        TARGET,
        {
            "squeeze_output_path": lambda source: "squeeze_output = X.ndim == 4" in source,
            "squeeze_return": lambda source: (
                "return Y.squeeze(-1) if squeeze_output else Y" in source
            ),
        },
    )


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
