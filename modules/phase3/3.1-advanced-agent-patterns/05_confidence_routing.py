"""
05 - Confidence-Based Routing
==============================
Route based on model confidence scores.

Key concept: Use confidence/logprobs to route uncertain cases to more powerful models or human review.

Book reference: AI_eng.10.3
"""

from openai import OpenAI
from pydantic import BaseModel
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


class Classification(BaseModel):
    """Job category classification."""
    category: str
    confidence: float


def classify_with_confidence(job: dict, model: str = "gpt-4o-mini") -> dict:
    """Classify job and get confidence score."""
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": """Classify this job into one category: engineering, design, product, sales, or operations.
Respond with JSON: {"category": "...", "confidence": 0.0-1.0}"""
            },
            {
                "role": "user",
                "content": f"Title: {job['title']}\nDescription: {job.get('description', '')[:200]}"
            }
        ],
        logprobs=True
    )

    # Parse response
    import json
    result = json.loads(response.output_text)

    # Calculate confidence from logprobs (simplified)
    # In production, you'd aggregate logprobs over the output tokens
    avg_logprob = -0.5  # Mock value - real impl would compute from response.logprobs
    confidence = min(result.get("confidence", 0.5), 0.99)

    return {
        "category": result["category"],
        "confidence": confidence,
        "model": model
    }


def route_by_confidence(
    job: dict,
    confidence_threshold: float = 0.7,
    fallback_model: str = "gpt-4o"
) -> dict:
    """Route to stronger model if confidence is low."""
    print(f"\nClassifying: {job['title'][:50]}...")

    # Try fast model first
    result = classify_with_confidence(job, "gpt-4o-mini")

    print(f"  Model: gpt-4o-mini")
    print(f"  Category: {result['category']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Route to stronger model if low confidence
    if result["confidence"] < confidence_threshold:
        print(f"  ⚠ Low confidence - routing to {fallback_model}")

        result = classify_with_confidence(job, fallback_model)

        print(f"  New confidence: {result['confidence']:.2f}")
        print(f"  Final category: {result['category']}")

    else:
        print(f"  ✓ High confidence - using result")

    return result


def batch_classify_with_routing(
    jobs: list[dict],
    confidence_threshold: float = 0.7
) -> dict:
    """Classify multiple jobs with confidence routing."""
    results = []
    model_usage = {"gpt-4o-mini": 0, "gpt-4o": 0}

    for job in jobs:
        result = route_by_confidence(job, confidence_threshold)
        results.append(result)
        model_usage[result["model"]] += 1

    return {
        "results": results,
        "model_usage": model_usage,
        "total": len(jobs)
    }


if __name__ == "__main__":
    print("=== CONFIDENCE-BASED ROUTING ===\n")

    jobs = load_sample_jobs(5)

    # Test different confidence thresholds
    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        print(f"\n{'=' * 70}")
        print(f"THRESHOLD: {threshold}")
        print("=" * 70)

        result = batch_classify_with_routing(jobs[:3], threshold)

        print(f"\n--- Summary ---")
        print(f"Total jobs: {result['total']}")
        print(f"gpt-4o-mini calls: {result['model_usage']['gpt-4o-mini']}")
        print(f"gpt-4o calls: {result['model_usage']['gpt-4o']}")
        print(f"Cost savings: {(result['model_usage']['gpt-4o-mini'] / result['total'] * 100):.0f}% used cheap model")

    print(f"\n{'=' * 70}")
    print("\nKEY INSIGHT: Higher thresholds = more fallback calls = higher accuracy + cost")
    print("Lower thresholds = more fast model usage = lower cost + potential accuracy loss")
