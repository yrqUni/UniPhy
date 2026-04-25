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
        if match is None:
            continue
        test_number = int(match.group(1))
        discovered.append((test_number, path.stem))
    if not discovered:
        raise ValueError("no numerical tests discovered")
    expected = list(range(1, len(discovered) + 1))
    observed = [number for number, _ in discovered]
    if observed != expected:
        raise ValueError(
            f"test numbering must be consecutive from 1: observed={observed}"
        )
    return discovered


def resolve_tests(requested):
    discovered = discover_tests()
    if not requested:
        return discovered
    selected = []
    by_id = {f"T{number:02d}": entry for number, entry in discovered}
    by_name = {name: (number, name) for number, name in discovered}
    seen = set()
    for name in requested:
        if name in by_id:
            entry = by_id[name]
        elif name in by_name:
            entry = by_name[name]
        else:
            raise ValueError(f"unknown test: {name}")
        if entry in seen:
            continue
        selected.append(entry)
        seen.add(entry)
    return selected


def get_git_value(args):
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def get_device_value():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def build_json_report(results, args):
    return {
        "commit": get_git_value(["git", "rev-parse", "HEAD"]),
        "branch": get_git_value(["git", "branch", "--show-current"]),
        "date": datetime.now().isoformat(),
        "node": socket.gethostname().split(".", maxsplit=1)[0],
        "device": get_device_value(),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
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
    selected = resolve_tests(args.tests)
    for test_number, module_name in selected:
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
        duration = time.time() - started
        results.append(
            {
                "id": test_id,
                "name": module_name,
                "status": status,
                "max_error": max_error,
                "duration_sec": duration,
                "log": str(Path(args.log_dir) / f"{test_id}_result.txt"),
                "detail": detail,
            }
        )
    pass_count = sum(entry["status"] == "PASS" for entry in results)
    fail_count = sum(entry["status"] == "FAIL" for entry in results)
    skip_count = sum(entry["status"] == "SKIP" for entry in results)
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    summary_path = Path(args.log_dir) / f"node_{hostname}_summary.txt"
    summary_lines = [
        (
            f"PASS: {pass_count}  FAIL: {fail_count}  SKIP: {skip_count}  "
            f"TOTAL: {len(results)}"
        )
    ]
    summary_lines.extend(
        (
            f"{entry['status']} {entry['id']} "
            f"max_error={format_max_error(entry['max_error'])}"
        )
        for entry in results
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(build_json_report(results, args), indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
