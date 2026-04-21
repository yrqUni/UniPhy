import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, write_result
else:
    from ..utils import REPO_DIR, write_result

TEST_ID = "T22"
TARGET = "Model/UniPhy/PScan.py"


def run():
    source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{TARGET}"],
        text=True,
    )
    has_docstring = "Run the parallel scan recurrence; D=1 and trailing width 1 are padded internally." in source
    passed = has_docstring
    max_err = 0.0 if passed else 1.0
    detail = f"pscan_docstring_present={has_docstring}"
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
