"""
01 - Create Eval Dataset
=========================
Build test cases from real data for evaluation.

Key concept: Good evaluation starts with good test data - representative, diverse, and well-labeled.

Book reference: AI_eng.4, AI_eng.8
"""

import json
from typing import Any
from pydantic import BaseModel
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

class EvalCase(BaseModel):
    """Single evaluation test case."""
    input: str
    expected_output: str
    metadata: dict[str, Any] = {}


def create_classification_dataset(jobs: list[dict], n_samples: int = 20) -> list[EvalCase]:
    """Create dataset for job category classification."""
    # Define categories based on job titles
    category_keywords = {
        "Engineering": ["engineer", "developer", "software", "backend", "frontend", "devops"],
        "Product": ["product manager", "product owner", "pm"],
        "Design": ["designer", "ux", "ui", "design"],
        "Data": ["data scientist", "data analyst", "data engineer", "ml engineer"],
        "Sales": ["sales", "account executive", "business development"],
    }

    eval_cases = []

    for job in jobs[:n_samples]:
        title = job["title"].lower()

        # Determine category
        category = "Other"
        for cat, keywords in category_keywords.items():
            if any(kw in title for kw in keywords):
                category = cat
                break

        eval_cases.append(EvalCase(
            input=job["title"],
            expected_output=category,
            metadata={
                "job_id": job.get("id"),
                "company": job.get("company", "Unknown")
            }
        ))

    return eval_cases


def create_extraction_dataset(jobs: list[dict], n_samples: int = 10) -> list[EvalCase]:
    """Create dataset for skill extraction."""
    common_skills = {
        "Python", "Java", "JavaScript", "React", "AWS", "Docker",
        "Kubernetes", "SQL", "MongoDB", "Git", "Agile", "REST API"
    }

    eval_cases = []

    for job in jobs[:n_samples]:
        description = job.get("description", "")

        # Find skills mentioned in description
        found_skills = [skill for skill in common_skills if skill.lower() in description.lower()]

        if found_skills:
            eval_cases.append(EvalCase(
                input=description[:500],
                expected_output=", ".join(sorted(found_skills)),
                metadata={
                    "job_id": job.get("id"),
                    "title": job["title"]
                }
            ))

    return eval_cases


def save_eval_dataset(eval_cases: list[EvalCase], filename: str):
    """Save evaluation dataset to JSON."""
    data = [case.model_dump() for case in eval_cases]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(eval_cases)} test cases to {filename}")


def load_eval_dataset(filename: str) -> list[EvalCase]:
    """Load evaluation dataset from JSON."""
    with open(filename, 'r') as f:
        data = json.load(f)

    return [EvalCase(**case) for case in data]


def analyze_dataset(eval_cases: list[EvalCase]):
    """Analyze dataset quality."""
    print("\n=== DATASET ANALYSIS ===\n")
    print(f"Total test cases: {len(eval_cases)}")

    # Count unique outputs
    outputs = [case.expected_output for case in eval_cases]
    unique_outputs = set(outputs)
    print(f"Unique outputs: {len(unique_outputs)}")

    # Distribution
    from collections import Counter
    distribution = Counter(outputs)

    print("\nOutput distribution:")
    for output, count in distribution.most_common():
        print(f"  {output}: {count}")

    # Check for imbalance
    max_count = max(distribution.values())
    min_count = min(distribution.values())
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("⚠️  Warning: Dataset is imbalanced - consider balancing")
    else:
        print("✓ Dataset is reasonably balanced")


if __name__ == "__main__":
    print("=== CREATING EVALUATION DATASET ===\n")

    # Load jobs
    jobs = load_sample_jobs(100)

    # Create classification dataset
    print("Creating classification dataset...")
    classification_cases = create_classification_dataset(jobs, n_samples=30)
    analyze_dataset(classification_cases)
    save_eval_dataset(classification_cases, "eval_classification.json")

    # Create extraction dataset
    print("\n" + "=" * 70)
    print("Creating extraction dataset...")
    extraction_cases = create_extraction_dataset(jobs, n_samples=20)
    print(f"\nCreated {len(extraction_cases)} extraction test cases")
    save_eval_dataset(extraction_cases, "eval_extraction.json")

    # Show samples
    print("\n" + "=" * 70)
    print("=== SAMPLE TEST CASES ===\n")
    print("Classification:")
    for case in classification_cases[:3]:
        print(f"  Input: {case.input}")
        print(f"  Expected: {case.expected_output}\n")

    print("\nKey insight: Quality eval data = reliable measurements")
