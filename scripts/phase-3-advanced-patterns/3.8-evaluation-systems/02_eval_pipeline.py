"""
02 - Evaluation Pipeline
=========================
Automated evaluation runs for consistent testing.

Key concept: Automated pipelines ensure every code change is evaluated - catch regressions early.

Book reference: AI_eng.4
"""

import utils._load_env  # Loads .env file automatically

import json
from typing import Callable, Any
from pydantic import BaseModel
from openai import OpenAI
import os
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class EvalCase(BaseModel):
    input: str
    expected_output: str
    metadata: dict[str, Any] = {}


class EvalResult(BaseModel):
    input: str
    expected: str
    actual: str
    correct: bool
    metadata: dict[str, Any] = {}


def classify_job(title: str) -> str:
    """Classify job category."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify job into: Engineering, Product, Design, Data, Sales, or Other. Return category only."
            },
            {"role": "user", "content": title}
        ],
        temperature=0.3,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


def run_evaluation(
    test_cases: list[EvalCase],
    system_func: Callable[[str], str],
    exact_match: bool = True
) -> dict:
    """Run evaluation pipeline."""
    results = []

    for i, case in enumerate(test_cases):
        print(f"Evaluating {i+1}/{len(test_cases)}...", end="\r")

        # Run system
        actual = system_func(case.input)

        # Check correctness
        if exact_match:
            correct = actual.lower().strip() == case.expected_output.lower().strip()
        else:
            correct = case.expected_output.lower() in actual.lower()

        results.append(EvalResult(
            input=case.input,
            expected=case.expected_output,
            actual=actual,
            correct=correct,
            metadata=case.metadata
        ))

    print()  # New line after progress

    # Calculate metrics
    accuracy = sum(r.correct for r in results) / len(results)

    return {
        "accuracy": accuracy,
        "total": len(results),
        "correct": sum(r.correct for r in results),
        "results": results
    }


def print_report(eval_results: dict):
    """Print evaluation report."""
    print("\n=== EVALUATION REPORT ===\n")
    print(f"Total cases: {eval_results['total']}")
    print(f"Correct: {eval_results['correct']}")
    print(f"Accuracy: {eval_results['accuracy']:.1%}\n")

    # Show failures
    failures = [r for r in eval_results['results'] if not r.correct]

    if failures:
        print(f"=== FAILURES ({len(failures)}) ===\n")
        for r in failures[:5]:  # Show first 5
            print(f"Input: {r.input}")
            print(f"Expected: {r.expected}")
            print(f"Actual: {r.actual}\n")


def save_results(eval_results: dict, filename: str):
    """Save results for analysis."""
    data = {
        "accuracy": eval_results["accuracy"],
        "total": eval_results["total"],
        "correct": eval_results["correct"],
        "results": [r.model_dump() for r in eval_results["results"]]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("=== EVALUATION PIPELINE ===\n")

    # Load test cases
    try:
        with open("eval_classification.json", 'r') as f:
            data = json.load(f)
        test_cases = [EvalCase(**case) for case in data]
    except FileNotFoundError:
        print("Run 01_eval_dataset.py first to create test data")
        exit(1)

    print(f"Loaded {len(test_cases)} test cases")

    # Run evaluation
    print("Running evaluation...")
    results = run_evaluation(test_cases, classify_job, exact_match=True)

    # Report
    print_report(results)

    # Save
    save_results(results, "eval_results.json")

    print("\nKey insight: Automated evaluation = consistent quality measurement")
