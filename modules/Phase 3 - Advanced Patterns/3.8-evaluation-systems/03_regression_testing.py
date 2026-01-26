"""
03 - Regression Testing
========================
Catch quality regressions when you change prompts or models.

Key concept: Compare current performance to baseline - flag any degradation before deployment.

Book reference: AI_eng.4
"""

import json
from datetime import datetime
from typing import Any
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

def load_results(filename: str) -> dict:
    """Load evaluation results."""
    with open(filename, 'r') as f:
        return json.load(f)


def compare_results(baseline: dict, current: dict) -> dict:
    """Compare current results to baseline."""
    baseline_acc = baseline["accuracy"]
    current_acc = current["accuracy"]

    delta = current_acc - baseline_acc
    delta_pct = (delta / baseline_acc) * 100 if baseline_acc > 0 else 0

    # Check for regressions
    regression_threshold = -0.05  # 5% drop is a regression

    status = "PASS"
    if delta < regression_threshold:
        status = "REGRESSION"
    elif delta > 0.05:
        status = "IMPROVEMENT"

    return {
        "baseline_accuracy": baseline_acc,
        "current_accuracy": current_acc,
        "delta": delta,
        "delta_pct": delta_pct,
        "status": status
    }


def analyze_new_failures(baseline: dict, current: dict) -> list[dict]:
    """Find test cases that now fail but passed before."""
    # Create lookup of baseline results
    baseline_correct = {}
    for r in baseline["results"]:
        baseline_correct[r["input"]] = r["correct"]

    # Find new failures
    new_failures = []
    for r in current["results"]:
        input_text = r["input"]
        was_correct = baseline_correct.get(input_text, False)
        is_correct = r["correct"]

        if was_correct and not is_correct:
            new_failures.append(r)

    return new_failures


def print_regression_report(comparison: dict, new_failures: list[dict]):
    """Print regression test report."""
    print("\n=== REGRESSION TEST REPORT ===\n")
    print(f"Baseline accuracy: {comparison['baseline_accuracy']:.1%}")
    print(f"Current accuracy: {comparison['current_accuracy']:.1%}")
    print(f"Delta: {comparison['delta']:+.1%} ({comparison['delta_pct']:+.1f}%)")
    print(f"\nStatus: {comparison['status']}")

    if comparison['status'] == "REGRESSION":
        print("⚠️  REGRESSION DETECTED - DO NOT DEPLOY")
    elif comparison['status'] == "IMPROVEMENT":
        print("✓ IMPROVEMENT - Good to go!")
    else:
        print("✓ No significant change")

    if new_failures:
        print(f"\n=== NEW FAILURES ({len(new_failures)}) ===\n")
        for r in new_failures[:5]:
            print(f"Input: {r['input']}")
            print(f"Expected: {r['expected']}")
            print(f"Got: {r['actual']}\n")


def save_baseline(results: dict, baseline_file: str = "eval_baseline.json"):
    """Save current results as new baseline."""
    with open(baseline_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved new baseline to {baseline_file}")


if __name__ == "__main__":
    print("=== REGRESSION TESTING ===\n")

    try:
        # Load baseline
        baseline = load_results("eval_baseline.json")
        print("Loaded baseline results")
    except FileNotFoundError:
        print("No baseline found. Run evaluation first and save as baseline:")
        print("  python 02_eval_pipeline.py")
        print("  cp eval_results.json eval_baseline.json")
        exit(1)

    try:
        # Load current results
        current = load_results("eval_results.json")
        print("Loaded current results\n")
    except FileNotFoundError:
        print("No current results. Run: python 02_eval_pipeline.py")
        exit(1)

    # Compare
    comparison = compare_results(baseline, current)
    new_failures = analyze_new_failures(baseline, current)

    # Report
    print_regression_report(comparison, new_failures)

    # Decision
    print("\n" + "=" * 70)
    if comparison['status'] == "REGRESSION":
        print("RECOMMENDATION: Investigate and fix before deploying")
    else:
        print("RECOMMENDATION: Safe to deploy")

    print("\nKey insight: Regression tests prevent quality degradation")
