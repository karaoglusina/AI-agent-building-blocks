"""
06 - Human Evaluation Design
=============================
Design effective human evaluation processes.

Key concept: Automated metrics don't capture everything - human evaluation is essential for quality, but must be structured carefully.

Book reference: AI_eng.4, hands_on_LLM.III.12
"""

import json
import random
from typing import Any
from pydantic import BaseModel
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


class HumanEvalTask(BaseModel):
    """Single task for human evaluation."""
    task_id: str
    input: str
    output_a: str
    output_b: str
    metadata: dict[str, Any] = {}


class HumanEvalRubric(BaseModel):
    """Evaluation rubric for human annotators."""
    criteria: list[dict[str, str]]
    scale: dict[str, int]
    instructions: str


def create_comparison_tasks(outputs_a: list[dict], outputs_b: list[dict], n_samples: int = 20) -> list[HumanEvalTask]:
    """Create side-by-side comparison tasks."""
    tasks = []

    # Sample randomly
    indices = random.sample(range(min(len(outputs_a), len(outputs_b))), min(n_samples, len(outputs_a), len(outputs_b)))

    for i, idx in enumerate(indices):
        task = HumanEvalTask(
            task_id=f"task_{i+1}",
            input=outputs_a[idx]["input"],
            output_a=outputs_a[idx]["output"],
            output_b=outputs_b[idx]["output"],
            metadata={"index": idx}
        )
        tasks.append(task)

    return tasks


def create_rubric() -> HumanEvalRubric:
    """Define evaluation rubric."""
    return HumanEvalRubric(
        criteria=[
            {
                "name": "Accuracy",
                "description": "Is the classification correct?"
            },
            {
                "name": "Confidence",
                "description": "How confident are you in your judgment?"
            }
        ],
        scale={
            "A is much better": 2,
            "A is slightly better": 1,
            "Both equal": 0,
            "B is slightly better": -1,
            "B is much better": -2
        },
        instructions="""For each task:
1. Read the input carefully
2. Evaluate both outputs
3. Choose which is better based on accuracy
4. Rate your confidence: Low, Medium, High"""
    )


def export_for_annotation(tasks: list[HumanEvalTask], rubric: HumanEvalRubric, filename: str = "human_eval.json"):
    """Export tasks in annotation-ready format."""
    data = {
        "rubric": rubric.model_dump(),
        "tasks": [task.model_dump() for task in tasks],
        "meta": {
            "total_tasks": len(tasks),
            "status": "pending"
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(tasks)} tasks to {filename}")


def analyze_agreement(annotations: list[dict]) -> dict:
    """Analyze inter-annotator agreement."""
    # Simple agreement calculation
    agreements = 0
    total_pairs = 0

    # Group by task_id
    by_task = {}
    for ann in annotations:
        task_id = ann["task_id"]
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(ann["rating"])

    # Calculate pairwise agreement
    for task_id, ratings in by_task.items():
        if len(ratings) >= 2:
            for i in range(len(ratings)):
                for j in range(i + 1, len(ratings)):
                    total_pairs += 1
                    if ratings[i] == ratings[j]:
                        agreements += 1

    agreement_rate = agreements / total_pairs if total_pairs > 0 else 0

    return {
        "total_annotations": len(annotations),
        "unique_tasks": len(by_task),
        "pairwise_agreement": agreement_rate,
        "interpretation": "High" if agreement_rate > 0.7 else "Moderate" if agreement_rate > 0.5 else "Low"
    }


def best_practices():
    """Print human evaluation best practices."""
    print("\n=== HUMAN EVALUATION BEST PRACTICES ===\n")

    practices = [
        "1. Clear Instructions: Write detailed guidelines with examples",
        "2. Training: Train annotators before the actual task",
        "3. Multiple Annotators: Use 2-3 annotators per task for reliability",
        "4. Blind Evaluation: Hide which system produced which output",
        "5. Random Sampling: Sample randomly to avoid bias",
        "6. Quality Checks: Include gold standard tasks to verify quality",
        "7. Rubric: Define clear evaluation criteria and scale",
        "8. Measure Agreement: Calculate inter-annotator agreement",
        "9. Iterative: Start small, refine instructions, then scale",
        "10. Compensation: Pay annotators fairly for their time"
    ]

    for practice in practices:
        print(practice)


if __name__ == "__main__":
    print("=== HUMAN EVALUATION DESIGN ===\n")

    # Simulate two systems' outputs
    outputs_a = [
        {"input": "Senior Python Developer", "output": "Engineering"},
        {"input": "Product Manager", "output": "Product"},
        {"input": "UX Designer", "output": "Design"},
    ]

    outputs_b = [
        {"input": "Senior Python Developer", "output": "Engineering"},
        {"input": "Product Manager", "output": "Product"},
        {"input": "UX Designer", "output": "Other"},  # Wrong
    ]

    # Create comparison tasks
    tasks = create_comparison_tasks(outputs_a, outputs_b, n_samples=3)

    print(f"Created {len(tasks)} comparison tasks\n")

    # Show sample task
    print("=== SAMPLE TASK ===")
    sample = tasks[0]
    print(f"Input: {sample.input}")
    print(f"Output A: {sample.output_a}")
    print(f"Output B: {sample.output_b}")
    print("\nQuestion: Which output is better?")

    # Create rubric
    print("\n" + "=" * 70)
    rubric = create_rubric()

    # Export
    export_for_annotation(tasks, rubric)

    # Best practices
    print("\n" + "=" * 70)
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Structured human eval provides gold standard quality assessment")
    print("Use for: qualitative assessment, edge cases, overall quality")
