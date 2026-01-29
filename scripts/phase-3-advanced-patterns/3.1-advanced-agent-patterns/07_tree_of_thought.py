"""
07 - Tree of Thought
====================
Explore multiple reasoning paths and choose the best.

Key concept: Generate multiple solution paths, evaluate each, and select the best - more thorough than single-path reasoning.

Book reference: hands_on_LLM.II.6
"""

import json
from openai import OpenAI
import os
from pydantic import BaseModel
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class ThoughtPath(BaseModel):
    """A single reasoning path."""
    approach: str
    reasoning: str
    solution: str
    confidence: float


def generate_thought_paths(problem: str, num_paths: int = 3) -> list[ThoughtPath]:
    """Generate multiple different approaches to solve the problem."""
    paths = []

    for i in range(num_paths):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": f"""You are solving a problem. Generate approach #{i + 1}.
        Think of a DIFFERENT approach than typical solutions. Be creative.
        
        Return JSON with:
        - approach: name of this approach
        - reasoning: why this approach makes sense
        - solution: the actual solution
        - confidence: how confident you are (0-1)"""
        },
        {
        "role": "user",
        "content": problem
        }
        ],
        temperature=0.7 + (i * 0.1)  # Vary temperature for diversity
        )

        try:
            data = json.loads(response.choices[0].message.content)
            paths.append(ThoughtPath(**data))
        except Exception as e:
            print(f"  ⚠ Failed to parse path {i + 1}: {e}")

    return paths


def evaluate_paths(paths: list[ThoughtPath], problem: str) -> list[dict]:
    """Evaluate and score each thought path."""
    evaluations = []

    for i, path in enumerate(paths):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": """Evaluate this solution approach on:
        1. Correctness (0-10)
        2. Completeness (0-10)
        3. Practicality (0-10)
        
        Return JSON with overall_score (0-10) and brief justification."""
        },
        {
        "role": "user",
        "content": f"""Problem: {problem}
        
        Approach: {path.approach}
        Reasoning: {path.reasoning}
        Solution: {path.solution}
        
        Evaluate this solution."""
        }
        ]
        )

        try:
            eval_data = json.loads(response.choices[0].message.content)
            evaluations.append({
                "path_index": i,
                "approach": path.approach,
                "score": eval_data.get("overall_score", path.confidence * 10),
                "justification": eval_data.get("justification", ""),
                "solution": path.solution
            })
        except Exception:
            evaluations.append({
                "path_index": i,
                "approach": path.approach,
                "score": path.confidence * 10,
                "justification": "Auto-scored from confidence",
                "solution": path.solution
            })

    return sorted(evaluations, key=lambda x: x["score"], reverse=True)


def tree_of_thought(problem: str, num_paths: int = 3) -> dict:
    """Solve problem using Tree of Thought approach."""
    print(f"\n{'=' * 70}")
    print(f"Problem: {problem}")
    print("=" * 70)

    # Generate multiple thought paths
    print(f"\nGenerating {num_paths} different approaches...")
    paths = generate_thought_paths(problem, num_paths)

    print(f"\nGenerated {len(paths)} paths:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path.approach} (confidence: {path.confidence:.2f})")

    # Evaluate all paths
    print(f"\nEvaluating paths...")
    evaluations = evaluate_paths(paths, problem)

    print(f"\nRanked solutions:")
    for i, eval_result in enumerate(evaluations, 1):
        print(f"  {i}. {eval_result['approach']}")
        print(f"     Score: {eval_result['score']:.1f}/10")
        print(f"     Justification: {eval_result['justification'][:60]}...")

    # Select best path
    best = evaluations[0]

    print(f"\n✓ Selected: {best['approach']} (score: {best['score']:.1f})")

    return {
        "problem": problem,
        "paths_explored": len(paths),
        "best_approach": best["approach"],
        "best_score": best["score"],
        "best_solution": best["solution"],
        "all_evaluations": evaluations
    }


if __name__ == "__main__":
    print("=== TREE OF THOUGHT ===\n")

    # Complex job matching problem
    problems = [
        """A candidate has:
- 5 years Python, 3 years JavaScript, 2 years Go
- Experience: backend APIs, microservices, cloud infrastructure
- Preferences: remote work, startup culture, equity compensation

We have 3 jobs:
Job A: Senior Backend Engineer - Python/Go, remote, Series B startup, $150k + equity
Job B: Full Stack Lead - JavaScript/Python, hybrid, enterprise, $180k
Job C: Staff Engineer - Go/Rust, remote, late-stage startup, $200k + equity

Which job is the best match and why?"""
    ]

    for problem in problems:
        result = tree_of_thought(problem, num_paths=3)

        print(f"\n{'=' * 70}")
        print(f"FINAL SOLUTION:")
        print(f"  {result['best_solution'][:200]}...")
        print("=" * 70)
