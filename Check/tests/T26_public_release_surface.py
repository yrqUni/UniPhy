import sys

from Check.utils import run_source_guard, write_result

TEST_ID = "T26"
TARGETS = {
    "README.md": {
        "no_private_path": lambda source: "/nfs/" not in source,
        "no_ssh_clone": lambda source: "git@github.com" not in source,
        "strict_stage2": lambda source: "strict-loads" in source,
    },
    "Check/README.md": {
        "no_private_path": lambda source: "/nfs/" not in source,
        "no_ssh_clone": lambda source: "git@github.com" not in source,
    },
    "Exp/ERA5/train.yaml": {
        "no_private_path": lambda source: "/nfs/" not in source,
    },
    "Exp/ERA5/align.yaml": {
        "no_private_path": lambda source: "/nfs/" not in source,
    },
    "Check/utils.py": {
        "no_private_path": lambda source: "/nfs/" not in source,
    },
    "Check/tests/run_all.py": {
        "no_unknown_commit": lambda source: '"UNKNOWN"' not in source,
        "no_internal_branch": lambda source: '"math-fixes"' not in source,
    },
}


def run():
    outcomes = []
    for relative_path, checks in TARGETS.items():
        status, _, detail = run_source_guard(TEST_ID, relative_path, checks)
        outcomes.append((status == "PASS", relative_path, detail))
    passed = all(item[0] for item in outcomes)
    detail = " ".join(f"{path}:{entry}" for _, path, entry in outcomes)
    status = "PASS" if passed else "FAIL"
    max_error = 0.0 if passed else 1.0
    return status, max_error, detail


if __name__ == "__main__":
    status, max_error, detail = run()
    write_result(TEST_ID, status, max_error, detail)
    sys.exit(0 if status == "PASS" else 1)
