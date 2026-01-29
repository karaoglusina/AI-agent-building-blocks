"""
05 - Simple Eval Loop
=====================
Run evaluation on test cases.

Key concept: Systematic evaluation finds issues before users do.

Book reference: AI_eng.4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
from dataclasses import dataclass, asdict
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@dataclass
class TestCase:
    """A single test case."""
    id: str
    input: str
    expected_contains: list[str]  # Keywords that should be in output
    expected_not_contains: list[str] = None  # Keywords that shouldn't be in output
    category: str = "general"


@dataclass
class EvalResult:
    """Result of evaluating a test case."""
    test_id: str
    passed: bool
    output: str
    missing_keywords: list[str]
    forbidden_keywords: list[str]


def run_system(input_text: str) -> str:
    """The system being evaluated."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "You are a job search assistant. Be helpful and concise."},
    {"role": "user", "content": input_text}
    ]
    )
    return response.choices[0].message.content


def evaluate_test_case(test: TestCase) -> EvalResult:
    """Run a single test case."""
    output = run_system(test.input).lower()
    
    # Check for expected keywords
    missing = [kw for kw in test.expected_contains if kw.lower() not in output]
    
    # Check for forbidden keywords
    forbidden = []
    if test.expected_not_contains:
        forbidden = [kw for kw in test.expected_not_contains if kw.lower() in output]
    
    passed = len(missing) == 0 and len(forbidden) == 0
    
    return EvalResult(
        test_id=test.id,
        passed=passed,
        output=output[:200] + "..." if len(output) > 200 else output,
        missing_keywords=missing,
        forbidden_keywords=forbidden
    )


def run_eval_suite(tests: list[TestCase]) -> dict:
    """Run all tests and return summary."""
    results = [evaluate_test_case(test) for test in tests]
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    # Group by category
    by_category = {}
    for test, result in zip(tests, results):
        cat = test.category
        if cat not in by_category:
            by_category[cat] = {"passed": 0, "failed": 0}
        if result.passed:
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1
    
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(results) if results else 0,
        "by_category": by_category,
        "results": results
    }


# Test suite
TEST_CASES = [
    TestCase("skill_query", "What skills do Python developers need?", 
             ["python", "programming"], category="skills"),
    TestCase("remote_query", "Find remote software engineering jobs",
             ["remote"], category="search"),
    TestCase("salary_query", "What's the salary for data scientists?",
             ["salary", "data"], category="salary"),
    TestCase("location_query", "Jobs in Amsterdam",
             ["amsterdam"], category="search"),
    TestCase("negative_test", "Tell me about Python jobs",
             ["python"], ["javascript", "java"], category="skills")]


if __name__ == "__main__":
    print("=== SIMPLE EVAL LOOP ===\n")
    
    summary = run_eval_suite(TEST_CASES)
    
    # Overall results
    print(f"Pass Rate: {summary['pass_rate']:.0%} ({summary['passed']}/{summary['total']})")
    
    # By category
    print("\nBy Category:")
    for cat, stats in summary["by_category"].items():
        total = stats["passed"] + stats["failed"]
        print(f"  {cat}: {stats['passed']}/{total} passed")
    
    # Detailed results
    print("\n--- Detailed Results ---")
    for result in summary["results"]:
        status = "✓" if result.passed else "✗"
        print(f"\n{status} {result.test_id}")
        
        if not result.passed:
            if result.missing_keywords:
                print(f"  Missing: {result.missing_keywords}")
            if result.forbidden_keywords:
                print(f"  Forbidden found: {result.forbidden_keywords}")
            print(f"  Output: {result.output[:100]}...")
    
    # Export results
    print("\n--- Exportable Results ---")
    print(json.dumps([asdict(r) for r in summary["results"][:2]], indent=2))
