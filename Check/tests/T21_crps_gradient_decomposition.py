import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Check.utils import REPO_DIR, write_result
else:
    from ..utils import REPO_DIR, write_result

TEST_ID = "T21"
TARGET = "Exp/ERA5/align.py"


def run():
    source = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "show", f"HEAD:{TARGET}"],
        text=True,
    )
    has_dead_pred_mean = "pred_mean = (pred_seq + other_sum) / ensemble_size" in source
    has_dead_delete = "del pred_mean" in source
    has_crps_comment = (
        "pairwise_i sums over j != k" in source
        and "0.5 * E|X-X'|" in source
    )
    passed = (not has_dead_pred_mean) and (not has_dead_delete) and has_crps_comment
    max_err = 0.0 if passed else 1.0
    detail = (
        f"dead_pred_mean={has_dead_pred_mean} dead_delete={has_dead_delete} "
        f"crps_comment={has_crps_comment}"
    )
    status = "PASS" if passed else "FAIL"
    return status, max_err, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
