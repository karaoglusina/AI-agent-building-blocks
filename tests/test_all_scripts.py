#!/usr/bin/env python3
"""
Smoke-test runner: executes every script with TEST_MODE=1 and reports pass/fail.

TEST_MODE=1 makes scripts skip their API calls, so this needs no OpenAI key. It does
need that phase's dependencies installed — TEST_MODE gates the network, not imports.

This is a SMOKE TEST. A pass means the script imported and ran to completion; it says
nothing about whether the output is correct. Bugs that only appear on a live API call
are invisible here — several such bugs (calling `.output_parsed` on a Chat Completions
response, validating against the wrong Pydantic model) survived in this repo for months
precisely because a green smoke test looked like proof. It catches rot, not wrongness.

Usage:
    python tests/test_all_scripts.py                      # every phase
    python tests/test_all_scripts.py --filter phase-1     # only matching paths
    python tests/test_all_scripts.py --list               # show what would run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


def is_test_file(name: str) -> bool:
    """True only for actual pytest-style files.

    Matching a bare "test" substring would exclude curriculum scripts that merely have
    it in the title — e.g. 03_regression_testing.py, which is a lesson, not a test.
    """
    stem = name.removesuffix(".py")
    return stem.startswith("test_") or stem.endswith("_test")


def find_all_scripts(scripts_dir: Path, name_filter: str = "") -> List[Path]:
    """Find all runnable scripts, optionally filtered by a substring of their path."""
    scripts = []
    for py_file in scripts_dir.rglob("*.py"):
        if py_file.name == "__init__.py" or is_test_file(py_file.name):
            continue
        if name_filter and name_filter not in str(py_file.relative_to(PROJECT_ROOT)):
            continue
        scripts.append(py_file)
    return sorted(scripts)


def run_script(script_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """Run one script under TEST_MODE and return (success, error_message)."""
    try:
        env = os.environ.copy()
        env["TEST_MODE"] = "1"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_path.parent,
            env=env,
        )
        if result.returncode == 0:
            return True, ""
        error_msg = result.stderr.strip() or result.stdout.strip()
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return False, error_msg
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 - report anything the runner hits
        return False, str(e)[:200]


def write_report(path: Path, results, passed: int, failed: int) -> None:
    with open(path, "w") as f:
        f.write("# Script Test Results\n\n")
        f.write(f"**Summary:** {passed} passed, {failed} failed out of {len(results)} total\n\n")
        f.write("| Script | Status | Error Message |\n")
        f.write("|--------|--------|---------------|\n")
        for script_path, status, error_msg in results:
            script_display = script_path.replace("|", "\\|")
            error_display = error_msg.replace("|", "\\|").replace("\n", " ")
            icon = "✓" if status == "PASSED" else "✗"
            f.write(f"| {script_display} | {icon} {status} | {error_display} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="", help="only run scripts whose path contains this")
    parser.add_argument("--list", action="store_true", help="list matching scripts and exit")
    parser.add_argument("--report", default="", help="write a markdown report to this path")
    args = parser.parse_args()

    scripts_dir = PROJECT_ROOT / "scripts"
    if not scripts_dir.exists():
        print(f"Error: scripts directory not found at {scripts_dir}")
        sys.exit(1)

    scripts = find_all_scripts(scripts_dir, args.filter)
    if not scripts:
        print(f"No scripts matched filter {args.filter!r}")
        sys.exit(1)

    if args.list:
        for s in scripts:
            print(s.relative_to(PROJECT_ROOT))
        return

    print(f"Running {len(scripts)} scripts with TEST_MODE=1\n" + "=" * 80)
    results, passed, failed = [], 0, 0

    for i, script_path in enumerate(scripts, 1):
        rel_path = script_path.relative_to(PROJECT_ROOT)
        print(f"[{i}/{len(scripts)}] {rel_path}", end=" ... ", flush=True)
        success, error_msg = run_script(script_path)
        if success:
            print("✓ PASSED")
            passed += 1
            results.append((str(rel_path), "PASSED", ""))
        else:
            print("✗ FAILED")
            failed += 1
            results.append((str(rel_path), "FAILED", error_msg))

    print("=" * 80)
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(scripts)} total\n")

    if failed:
        print("Failures:")
        for script_path, status, error_msg in results:
            if status == "FAILED":
                print(f"  ✗ {script_path}\n      {error_msg.splitlines()[0] if error_msg else ''}")
        print()

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report_path, results, passed, failed)
        print(f"Results saved to: {report_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
