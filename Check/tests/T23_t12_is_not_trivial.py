import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, write_result
else:
    from ..utils import REPO_DIR, write_result

TEST_ID = "T23"
TARGET = "Check/tests/T12_basis_biorthogonality.py"


def run():
    source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{TARGET}"],
        text=True,
    )
    has_constant_pass = 'return "PASS", residual_100, detail' in source
    has_broken_case = "residual_broken" in source
    has_status_gate = 'status = "PASS" if passed else "FAIL"' in source
    passed = (not has_constant_pass) and has_broken_case and has_status_gate
    max_err = 0.0 if passed else 1.0
    detail = (
        f"constant_pass={has_constant_pass} "
        f"broken_case={has_broken_case} status_gate={has_status_gate}"
    )
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
