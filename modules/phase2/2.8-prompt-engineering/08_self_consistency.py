"""
08 - Self-Consistency
=====================
Sample multiple outputs and aggregate.

Key concept: Multiple samples with voting improves reliability on hard problems.

Book reference: hands_on_LLM.II.6
"""

from openai import OpenAI
from collections import Counter

client = OpenAI()


def get_single_answer(question: str, temperature: float = 0.7) -> str:
    """Get a single answer with some randomness."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Answer the question. Give only the final answer, no explanation."
            },
            {"role": "user", "content": question}
        ],
        temperature=temperature
    )
    return response.output_text.strip()


def self_consistent_answer(question: str, n_samples: int = 5) -> tuple[str, dict]:
    """Get multiple samples and return the most common answer."""
    answers = []
    
    for _ in range(n_samples):
        answer = get_single_answer(question, temperature=0.7)
        answers.append(answer)
    
    # Count answers (normalize to lowercase for comparison)
    normalized = [a.lower().strip() for a in answers]
    counter = Counter(normalized)
    
    # Get most common
    most_common, count = counter.most_common(1)[0]
    confidence = count / n_samples
    
    # Return original casing version
    for a in answers:
        if a.lower().strip() == most_common:
            return a, {"answers": answers, "confidence": confidence, "counts": dict(counter)}
    
    return most_common, {"answers": answers, "confidence": confidence, "counts": dict(counter)}


def classify_with_consistency(text: str, categories: list[str], n_samples: int = 5) -> tuple[str, float]:
    """Classify with self-consistency voting."""
    prompt = f"Classify this text into one of: {', '.join(categories)}. Reply with only the category name.\n\nText: {text}"
    
    answer, stats = self_consistent_answer(prompt, n_samples)
    
    # Map to closest category
    for cat in categories:
        if cat.lower() in answer.lower():
            return cat, stats["confidence"]
    
    return answer, stats["confidence"]


if __name__ == "__main__":
    print("=== SELF-CONSISTENCY ===\n")
    
    # Reasoning question where answers might vary
    question = "A job posting requires 3+ years Python OR 5+ years Java. A candidate has 2 years Python and 4 years Java. Are they qualified? Answer Yes or No."
    
    print(f"Question: {question}\n")
    
    # Single sample (less reliable)
    print("--- Single Sample ---")
    for i in range(3):
        answer = get_single_answer(question)
        print(f"  Sample {i+1}: {answer}")
    
    # Self-consistent (more reliable)
    print("\n--- Self-Consistent (5 samples) ---")
    final_answer, stats = self_consistent_answer(question, n_samples=5)
    
    print(f"All answers: {stats['answers']}")
    print(f"Answer counts: {stats['counts']}")
    print(f"Final answer: {final_answer} (confidence: {stats['confidence']:.0%})")
    
    # Classification example
    print("\n\n=== Classification with Self-Consistency ===")
    texts = [
        "Looking for a senior engineer to build ML pipelines",
        "We need a creative designer for our mobile app",
        "Sales representative to grow our enterprise accounts",
    ]
    categories = ["Engineering", "Design", "Sales", "Marketing", "Operations"]
    
    for text in texts:
        cat, conf = classify_with_consistency(text, categories, n_samples=5)
        print(f"\n\"{text[:50]}...\"")
        print(f"  Category: {cat} (confidence: {conf:.0%})")
