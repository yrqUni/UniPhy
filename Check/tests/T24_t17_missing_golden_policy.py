import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, write_result
else:
    from ..utils import REPO_DIR, write_result

TEST_ID = "T24"
TARGET = "Check/tests/T17_numerical_regression.py"


def run():
    source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{TARGET}"],
        text=True,
    )
    has_autosave = 'torch.save({"fwd": out_fwd_r.cpu(), "roll": out_roll_r.cpu()}, GOLDEN_PATH)' in source
    has_pass_on_missing = 'return "PASS", 0.0, "golden values saved"' in source
    has_skip_on_missing = 'return "SKIP"' in source
    passed = (not has_autosave) and (not has_pass_on_missing) and has_skip_on_missing
    max_err = 0.0 if passed else 1.0
    detail = (
        f"autosave={has_autosave} pass_on_missing={has_pass_on_missing} "
        f"skip_on_missing={has_skip_on_missing}"
    )
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
