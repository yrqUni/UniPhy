import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, write_result
else:
    from ..utils import REPO_DIR, write_result

TEST_ID = "T25"
RUNNER = "Check/tests/run_all.py"
T17 = "Check/tests/T17_numerical_regression.py"


def run():
    runner_source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{RUNNER}"],
        text=True,
    )
    t17_source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{T17}"],
        text=True,
    )
    has_json_out = '--json-out' in runner_source
    has_json_write = 'json.dumps(build_json_report(results, args)' in runner_source
    has_regen_flag = '--regenerate' in t17_source
    has_sha256 = 'sha256=' in t17_source
    passed = has_json_out and has_json_write and has_regen_flag and has_sha256
    max_err = 0.0 if passed else 1.0
    detail = (
        f"json_out={has_json_out} json_write={has_json_write} "
        f"regen_flag={has_regen_flag} sha256={has_sha256}"
    )
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
