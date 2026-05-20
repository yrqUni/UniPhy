import importlib
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from Check.utils import LOG_DIR, format_max_error, write_result

TEST_PATTERN = re.compile(r"^T(\d+)_.*\.py$")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def discover_tests():
    tests_dir = Path(__file__).resolve().parent
    discovered = []
    for path in sorted(tests_dir.glob("T*.py")):
        match = TEST_PATTERN.match(path.name)
        if match is not None:
            discovered.append((int(match.group(1)), path.stem))
    if not discovered:
        raise ValueError("no numerical tests discovered")
    observed = [number for number, _ in discovered]
    expected = list(range(1, len(discovered) + 1))
    if observed != expected:
        raise ValueError(f"test numbering must be consecutive from 1: observed={observed}")
    return discovered


def resolve_tests(requested):
    discovered = discover_tests()
    if not requested:
        return discovered
    by_id = {f"T{number:02d}": (number, name) for number, name in discovered}
    by_name = {name: (number, name) for number, name in discovered}
    selected = []
    seen = set()
    for item in requested:
        entry = by_id.get(item, by_name.get(item))
        if entry is None:
            raise ValueError(f"unknown test: {item}")
        if entry not in seen:
            selected.append(entry)
            seen.add(entry)
    return selected


def git_value(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def build_json_report(results, args):
    return {
        "commit": git_value(["git", "rev-parse", "HEAD"]),
        "branch": git_value(["git", "branch", "--show-current"]),
        "date": datetime.now().isoformat(),
        "node": socket.gethostname().split(".", maxsplit=1)[0],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "env": {"python": sys.version.split()[0], "torch": torch.__version__},
        "summary": {
            "total": len(results),
            "pass": sum(entry["status"] == "PASS" for entry in results),
            "fail": sum(entry["status"] == "FAIL" for entry in results),
            "skip": sum(entry["status"] == "SKIP" for entry in results),
        },
        "tests": results,
        "log_dir": str(Path(args.log_dir)),
    }


def main():
    args = parse_args()
    results = []
    for test_number, module_name in resolve_tests(args.tests):
        module = importlib.import_module(f"Check.tests.{module_name}")
        started = time.time()
        test_id = f"T{test_number:02d}"
        try:
            status, max_error, detail = module.run()
            write_result(test_id, status, max_error, detail, log_dir=args.log_dir)
        except Exception as exc:
            status = "FAIL"
            max_error = "-"
            detail = f"{type(exc).__name__}: {exc}"
            write_result(test_id, status, max_error, detail, log_dir=args.log_dir)
        results.append(
            {
                "id": test_id,
                "name": module_name,
                "status": status,
                "max_error": max_error,
                "duration_sec": time.time() - started,
                "log": str(Path(args.log_dir) / f"{test_id}_result.txt"),
                "detail": detail,
            }
        )
    pass_count = sum(entry["status"] == "PASS" for entry in results)
    fail_count = sum(entry["status"] == "FAIL" for entry in results)
    skip_count = sum(entry["status"] == "SKIP" for entry in results)
    summary_path = Path(args.log_dir) / f"node_{socket.gethostname().split('.', maxsplit=1)[0]}_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  TOTAL: {len(results)}"]
    lines.extend(f"{entry['status']} {entry['id']} max_error={format_max_error(entry['max_error'])}" for entry in results)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(build_json_report(results, args), indent=2) + "\n", encoding="utf-8")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
