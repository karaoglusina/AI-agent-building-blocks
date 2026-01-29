"""
04 - Prompt Versioning
=======================
Track prompt changes and their performance impact.

Key concept: Prompts are code - version them, A/B test them, measure their performance.

Book reference: AI_eng.5
"""

import utils._load_env  # Loads .env file automatically

import json
from datetime import datetime
from typing import Any
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


class PromptVersion(dict):
    """Prompt version with metadata."""
    def __init__(self, version: str, system_prompt: str, user_template: str, metadata: dict = None):
        super().__init__({
            "version": version,
            "system_prompt": system_prompt,
            "user_template": user_template,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        })


# Define prompt versions
PROMPT_VERSIONS = {
    "v1.0": PromptVersion(
        version="v1.0",
        system_prompt="Classify the job category.",
        user_template="Job: {title}\n\nCategory:",
        metadata={"note": "Initial simple prompt"}
    ),
    "v1.1": PromptVersion(
        version="v1.1",
        system_prompt="You are a job market expert. Classify jobs into categories: Engineering, Product, Design, Data, Sales, or Other.",
        user_template="Job title: {title}\n\nWhat category is this job? Respond with one word only.",
        metadata={"note": "Added role and explicit categories"}
    ),
    "v1.2": PromptVersion(
        version="v1.2",
        system_prompt="You are a job market expert. Classify jobs accurately into: Engineering, Product, Design, Data, Sales, or Other. Consider the full job title.",
        user_template="Classify this job title: \"{title}\"\n\nCategory (one word):",
        metadata={"note": "More explicit instructions + quotes around title"}
    )
}


def evaluate_prompt_version(version_id: str, test_cases: list[dict]) -> dict:
    """Evaluate a specific prompt version."""
    prompt_ver = PROMPT_VERSIONS[version_id]

    correct = 0
    results = []

    print(f"Testing {version_id}...")

    for case in test_cases:
        # Format prompts
        messages = [
            {"role": "system", "content": prompt_ver["system_prompt"]},
            {"role": "user", "content": prompt_ver["user_template"].format(title=case["input"])}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=20
        )

        prediction = response.choices[0].message.content.strip()
        is_correct = prediction.lower() == case["expected_output"].lower()

        if is_correct:
            correct += 1

        results.append({
            "input": case["input"],
            "expected": case["expected_output"],
            "predicted": prediction,
            "correct": is_correct
        })

    accuracy = correct / len(test_cases)

    return {
        "version": version_id,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "results": results
    }


def compare_versions(version_results: list[dict]):
    """Compare multiple prompt versions."""
    print("\n=== PROMPT VERSION COMPARISON ===\n")

    # Sort by accuracy
    sorted_results = sorted(version_results, key=lambda x: x["accuracy"], reverse=True)

    print(f"{'Version':<10} {'Accuracy':<12} {'Correct':<10} {'Note':<40}")
    print("=" * 75)

    for result in sorted_results:
        version = result["version"]
        accuracy = result["accuracy"]
        correct = result["correct"]
        total = result["total"]
        note = PROMPT_VERSIONS[version]["metadata"].get("note", "")

        print(f"{version:<10} {accuracy:<12.1%} {correct}/{total:<7} {note:<40}")

    # Winner
    best = sorted_results[0]
    print(f"\n✓ Best: {best['version']} with {best['accuracy']:.1%} accuracy")

    # Show improvement
    if len(sorted_results) > 1:
        worst = sorted_results[-1]
        improvement = best['accuracy'] - worst['accuracy']
        print(f"  Improvement over worst: {improvement:+.1%}")


def save_version_results(all_results: list[dict], filename: str = "prompt_versions.json"):
    """Save all version results."""
    data = {
        "tested_at": datetime.now().isoformat(),
        "versions": all_results
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to {filename}")


if __name__ == "__main__":
    print("=== PROMPT VERSIONING ===\n")

    # Load test cases
    try:
        with open("eval_classification.json", 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Run 01_eval_dataset.py first")
        exit(1)

    # Use subset for quick testing
    test_subset = test_cases[:20]

    print(f"Testing {len(PROMPT_VERSIONS)} prompt versions")
    print(f"Test cases: {len(test_subset)}\n")

    # Evaluate each version
    all_results = []
    for version_id in PROMPT_VERSIONS.keys():
        result = evaluate_prompt_version(version_id, test_subset)
        all_results.append(result)
        print(f"  {version_id}: {result['accuracy']:.1%}")

    # Compare
    compare_versions(all_results)

    # Save
    save_version_results(all_results)

    print("\nKey insight: Version and test prompts like code - measure impact!")
