"""
01 - When to Fine-tune
======================
Decision framework for choosing fine-tuning vs RAG vs prompt engineering.

Key concept: Fine-tuning is powerful but expensive. Use it when prompt engineering and RAG aren't sufficient for your specific task or domain.

Book reference: AI_eng.7
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class UseCase:
    """Represents a potential use case for LLM customization."""
    name: str
    description: str
    data_size: int  # Number of examples available
    has_labels: bool
    needs_consistency: bool
    latency_sensitive: bool
    cost_sensitive: bool


def evaluate_approach(use_case: UseCase) -> Tuple[str, str]:
    """
    Evaluate which approach is best for a given use case.

    Returns: (recommended_approach, reasoning)
    """
    score_prompt = 0
    score_rag = 0
    score_finetune = 0

    reasons = []

    # Data availability
    if use_case.data_size < 50:
        score_prompt += 3
        reasons.append("Limited data: Prompt engineering requires no training data")
    elif use_case.data_size < 500:
        score_prompt += 2
        score_rag += 2
        reasons.append("Moderate data: Good for RAG or careful prompting")
    else:
        score_finetune += 2
        reasons.append("Large dataset: Fine-tuning can leverage many examples")

    # Labeled data
    if not use_case.has_labels:
        score_prompt += 2
        score_rag += 2
        reasons.append("No labels: Unsupervised approaches preferred")
    else:
        score_finetune += 2
        reasons.append("Labeled data: Can train supervised model")

    # Consistency needs
    if use_case.needs_consistency:
        score_finetune += 3
        reasons.append("Needs consistency: Fine-tuning produces more reliable outputs")
    else:
        score_prompt += 1

    # Latency requirements
    if use_case.latency_sensitive:
        score_finetune += 2
        reasons.append("Latency sensitive: Fine-tuned models can be smaller/faster")

    # Cost considerations
    if use_case.cost_sensitive:
        score_prompt += 2
        reasons.append("Cost sensitive: Prompting has no training costs")
    else:
        score_finetune += 1

    # Determine best approach
    scores = {
        "Prompt Engineering": score_prompt,
        "RAG": score_rag,
        "Fine-tuning": score_finetune
    }

    best_approach = max(scores, key=scores.get)
    reasoning = " | ".join(reasons)

    return best_approach, reasoning


def when_to_use_prompt_engineering():
    """Show scenarios where prompt engineering is the best choice."""
    print("=== WHEN TO USE PROMPT ENGINEERING ===\n")

    scenarios = [
        "✓ Limited data (<50 examples)",
        "✓ Rapid prototyping and iteration",
        "✓ Simple task (summarization, Q&A)",
        "✓ Task requires reasoning or creativity",
        "✓ No budget for training",
        "✓ Need to change behavior quickly",
        "✓ General-purpose task (not domain-specific)",
    ]

    print("Best when:")
    for scenario in scenarios:
        print(f"  {scenario}")

    print("\nExample tasks:")
    print("  - General Q&A chatbot")
    print("  - Text summarization")
    print("  - Creative writing assistance")
    print("  - Few-shot classification")


def when_to_use_rag():
    """Show scenarios where RAG is the best choice."""
    print("\n" + "=" * 70)
    print("=== WHEN TO USE RAG ===\n")

    scenarios = [
        "✓ Need up-to-date information",
        "✓ Large knowledge base to query",
        "✓ Information changes frequently",
        "✓ Need source attribution",
        "✓ Domain knowledge in documents",
        "✓ Don't have labeled training data",
        "✓ Need interpretability (show sources)",
    ]

    print("Best when:")
    for scenario in scenarios:
        print(f"  {scenario}")

    print("\nExample tasks:")
    print("  - Document Q&A (contracts, manuals)")
    print("  - Customer support with knowledge base")
    print("  - Research assistant")
    print("  - Internal company chatbot")


def when_to_use_finetuning():
    """Show scenarios where fine-tuning is the best choice."""
    print("\n" + "=" * 70)
    print("=== WHEN TO USE FINE-TUNING ===\n")

    scenarios = [
        "✓ Need consistent output format/style",
        "✓ Domain-specific language (legal, medical)",
        "✓ Have 500+ high-quality labeled examples",
        "✓ Task not well-served by base model",
        "✓ Need lower latency (can use smaller model)",
        "✓ Need to reduce token usage in prompts",
        "✓ Want to teach specific behavior/tone",
        "✓ Complex structured output required",
    ]

    print("Best when:")
    for scenario in scenarios:
        print(f"  {scenario}")

    print("\nExample tasks:")
    print("  - Custom code generation (company style)")
    print("  - Medical diagnosis from clinical notes")
    print("  - Legal contract analysis")
    print("  - Structured data extraction")
    print("  - Custom chatbot personality")


def evaluate_real_world_cases():
    """Evaluate real-world use cases."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD CASE EVALUATION ===\n")

    cases = [
        UseCase(
            name="Customer Support Chatbot",
            description="Answer questions from company knowledge base",
            data_size=100,
            has_labels=False,
            needs_consistency=False,
            latency_sensitive=False,
            cost_sensitive=True
        ),
        UseCase(
            name="Legal Contract Classifier",
            description="Classify contract clauses into 20 categories",
            data_size=5000,
            has_labels=True,
            needs_consistency=True,
            latency_sensitive=False,
            cost_sensitive=False
        ),
        UseCase(
            name="Code Review Assistant",
            description="Suggest improvements to code",
            data_size=30,
            has_labels=False,
            needs_consistency=False,
            latency_sensitive=False,
            cost_sensitive=True
        ),
        UseCase(
            name="Medical Report Generator",
            description="Generate consistent clinical reports",
            data_size=2000,
            has_labels=True,
            needs_consistency=True,
            latency_sensitive=False,
            cost_sensitive=False
        ),
    ]

    for case in cases:
        approach, reasoning = evaluate_approach(case)
        print(f"Case: {case.name}")
        print(f"  Description: {case.description}")
        print(f"  Data: {case.data_size} examples | Labels: {case.has_labels}")
        print(f"  → Recommended: {approach}")
        print(f"  → Reasoning: {reasoning}")
        print()


def combination_approaches():
    """Show how to combine multiple approaches."""
    print("=" * 70)
    print("=== COMBINATION APPROACHES ===\n")

    print("You can often combine approaches for better results:\n")

    combinations = [
        ("RAG + Prompt Engineering",
         "Use RAG to retrieve context, then prompt to process it"),

        ("Fine-tuning + RAG",
         "Fine-tune for output format, RAG for up-to-date content"),

        ("Prompt Engineering → Fine-tuning",
         "Start with prompts, collect data, then fine-tune"),

        ("Fine-tuning + Prompt Engineering",
         "Fine-tune base behavior, prompt for specific requests"),
    ]

    for combo, description in combinations:
        print(f"{combo}")
        print(f"  → {description}\n")


def decision_tree():
    """Print a decision tree for approach selection."""
    print("=" * 70)
    print("=== DECISION TREE ===\n")

    print("""
    START
      │
      ├─ Do you have <50 examples?
      │   └─ YES → Prompt Engineering
      │
      ├─ Do you need up-to-date information?
      │   └─ YES → RAG (+ Prompt Engineering)
      │
      ├─ Do you have 500+ labeled examples?
      │   │
      │   ├─ YES → Do you need consistent output format?
      │   │   │
      │   │   ├─ YES → Fine-tuning
      │   │   └─ NO → Prompt Engineering or RAG
      │   │
      │   └─ NO → Prompt Engineering or RAG
      │
      └─ Is the base model failing at your task?
          └─ YES → Consider Fine-tuning
    """)


if __name__ == "__main__":
    when_to_use_prompt_engineering()
    when_to_use_rag()
    when_to_use_finetuning()
    evaluate_real_world_cases()
    combination_approaches()
    decision_tree()

    print("\n" + "=" * 70)
    print("\nKey insight: Start simple (prompting), add complexity (RAG),")
    print("then fine-tune only when needed. Most problems don't need fine-tuning!")
