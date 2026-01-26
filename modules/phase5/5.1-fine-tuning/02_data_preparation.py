"""
02 - Training Data Preparation
===============================
Prepare high-quality training data for fine-tuning.

Key concept: Data quality matters more than quantity. Clean, diverse, and well-formatted data produces better fine-tuned models.

Book reference: AI_eng.7, AI_eng.8
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import json
from typing import List, Dict, Any
from collections import Counter


def format_for_chat_finetuning(
    system_prompt: str,
    user_message: str,
    assistant_response: str
) -> Dict[str, Any]:
    """
    Format a training example for chat model fine-tuning.

    This follows OpenAI's fine-tuning format.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def format_training_data_examples():
    """Show how to format training data correctly."""
    print("=== TRAINING DATA FORMAT ===\n")

    # Example 1: Customer support
    example1 = format_for_chat_finetuning(
        system_prompt="You are a helpful customer support agent for TechCo.",
        user_message="My order hasn't arrived yet. Order #12345",
        assistant_response="I'll help you track order #12345. Let me look that up for you. According to our system, your order was shipped on Jan 20th and should arrive by Jan 25th. The tracking number is TRACK123. Is there anything else I can help with?"
    )

    print("Example 1: Customer Support")
    print(json.dumps(example1, indent=2))

    # Example 2: Code generation
    example2 = format_for_chat_finetuning(
        system_prompt="You are a Python expert. Generate clean, well-documented code.",
        user_message="Write a function to calculate fibonacci numbers",
        assistant_response='```python\ndef fibonacci(n: int) -> int:\n    """Calculate the nth Fibonacci number.\n    \n    Args:\n        n: Position in Fibonacci sequence (0-indexed)\n    \n    Returns:\n        The nth Fibonacci number\n    """\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```'
    )

    print("\nExample 2: Code Generation")
    print(json.dumps(example2, indent=2))


def assess_data_quality(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assess the quality of training data.

    Returns metrics about the dataset.
    """
    if not examples:
        return {"error": "No examples provided"}

    # Length statistics
    user_lengths = []
    assistant_lengths = []

    for example in examples:
        messages = example.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                user_lengths.append(len(msg["content"]))
            elif msg["role"] == "assistant":
                assistant_lengths.append(len(msg["content"]))

    # Calculate metrics
    metrics = {
        "total_examples": len(examples),
        "avg_user_length": sum(user_lengths) / len(user_lengths) if user_lengths else 0,
        "avg_assistant_length": sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0,
        "min_user_length": min(user_lengths) if user_lengths else 0,
        "max_user_length": max(user_lengths) if user_lengths else 0,
        "min_assistant_length": min(assistant_lengths) if assistant_lengths else 0,
        "max_assistant_length": max(assistant_lengths) if assistant_lengths else 0,
    }

    return metrics


def data_quality_checks():
    """Demonstrate data quality assessment."""
    print("\n" + "=" * 70)
    print("=== DATA QUALITY CHECKS ===\n")

    # Good dataset
    good_examples = [
        format_for_chat_finetuning(
            "You are a helpful assistant.",
            "What's the capital of France?",
            "The capital of France is Paris."
        ),
        format_for_chat_finetuning(
            "You are a helpful assistant.",
            "Explain photosynthesis briefly.",
            "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."
        ),
        format_for_chat_finetuning(
            "You are a helpful assistant.",
            "How do I sort a list in Python?",
            "You can sort a list in Python using the sorted() function or the .sort() method. Example: sorted([3,1,2]) returns [1,2,3]."
        ),
    ]

    metrics = assess_data_quality(good_examples)

    print("Dataset metrics:")
    print(f"  Total examples: {metrics['total_examples']}")
    print(f"  Avg user message length: {metrics['avg_user_length']:.0f} chars")
    print(f"  Avg assistant response length: {metrics['avg_assistant_length']:.0f} chars")
    print(f"  User length range: {metrics['min_user_length']}-{metrics['max_user_length']}")
    print(f"  Assistant length range: {metrics['min_assistant_length']}-{metrics['max_assistant_length']}")


def common_data_quality_issues():
    """Show common data quality problems and solutions."""
    print("\n" + "=" * 70)
    print("=== COMMON DATA QUALITY ISSUES ===\n")

    issues = [
        ("Too few examples (<100)",
         "Solution: Collect more data or use data augmentation"),

        ("Inconsistent formatting",
         "Solution: Standardize format, validate all examples"),

        ("Low-quality responses",
         "Solution: Review and clean data, use human evaluation"),

        ("Imbalanced categories",
         "Solution: Oversample minority classes, collect more diverse data"),

        ("Duplicates",
         "Solution: Deduplicate based on input/output similarity"),

        ("Too short/long examples",
         "Solution: Filter outliers, split long examples"),

        ("Noisy or incorrect labels",
         "Solution: Multi-rater validation, confidence scoring"),

        ("Train/test leakage",
         "Solution: Careful splitting, check for near-duplicates"),
    ]

    for issue, solution in issues:
        print(f"Issue: {issue}")
        print(f"  → {solution}\n")


def data_synthesis_techniques():
    """Show techniques for synthesizing training data."""
    print("=" * 70)
    print("=== DATA SYNTHESIS TECHNIQUES ===\n")

    print("1. Paraphrasing")
    print("   Original: 'What's the capital of France?'")
    print("   Paraphrased: 'Tell me the capital city of France'")
    print("   Paraphrased: 'Which city is the capital of France?'\n")

    print("2. Template-based generation")
    print("   Template: 'What is the [attribute] of [entity]?'")
    print("   Generated: 'What is the capital of France?'")
    print("   Generated: 'What is the population of Tokyo?'\n")

    print("3. LLM-assisted generation")
    print("   Prompt LLM: 'Generate 10 variations of: What's the weather?'")
    print("   - How's the weather today?")
    print("   - What's the forecast?")
    print("   - Tell me about today's weather\n")

    print("4. Back-translation")
    print("   English → French → English")
    print("   'What's the capital?' → 'Quelle est la capitale?' → 'What is the capital?'\n")

    print("5. Data augmentation")
    print("   - Add typos (for robustness)")
    print("   - Change entities (Paris → London)")
    print("   - Mix question formats")


def validation_split_strategy():
    """Show how to split data for training and validation."""
    print("\n" + "=" * 70)
    print("=== TRAIN/VALIDATION SPLIT ===\n")

    print("Recommended splits:")
    print("  - Small dataset (<1000): 80/20 train/val")
    print("  - Medium dataset (1000-10000): 85/15 train/val")
    print("  - Large dataset (>10000): 90/10 train/val\n")

    print("Best practices:")
    print("  ✓ Stratify by category (if classification)")
    print("  ✓ Check for near-duplicates across splits")
    print("  ✓ Ensure validation represents real distribution")
    print("  ✓ Hold out test set for final evaluation")
    print("  ✓ Use temporal split for time-series data\n")

    # Example split
    total_examples = 1000
    train_size = int(0.85 * total_examples)
    val_size = total_examples - train_size

    print(f"Example with {total_examples} examples:")
    print(f"  Training: {train_size} examples (85%)")
    print(f"  Validation: {val_size} examples (15%)")


def data_preparation_checklist():
    """Print a checklist for data preparation."""
    print("\n" + "=" * 70)
    print("=== DATA PREPARATION CHECKLIST ===\n")

    checklist = [
        "□ Collect sufficient examples (500+ recommended)",
        "□ Ensure consistent formatting",
        "□ Validate all examples are correct",
        "□ Remove duplicates",
        "□ Balance categories (if applicable)",
        "□ Check for bias in training data",
        "□ Anonymize sensitive information (PII)",
        "□ Split into train/validation sets",
        "□ Document data collection process",
        "□ Review sample of data manually",
        "□ Test format with small fine-tuning run",
        "□ Store data securely with backups",
    ]

    for item in checklist:
        print(f"  {item}")


def save_training_data_example():
    """Show how to save training data in correct format."""
    print("\n" + "=" * 70)
    print("=== SAVING TRAINING DATA ===\n")

    examples = [
        format_for_chat_finetuning(
            "You are a helpful assistant.",
            "What's 2+2?",
            "2+2 equals 4."
        ),
        format_for_chat_finetuning(
            "You are a helpful assistant.",
            "Name a primary color.",
            "Red is a primary color."
        ),
    ]

    # JSONL format (one JSON per line)
    print("Format: JSONL (JSON Lines)")
    print("Each line is a separate JSON object:\n")

    for example in examples:
        print(json.dumps(example))

    print("\nSave to file:")
    print('  with open("training_data.jsonl", "w") as f:')
    print('      for example in examples:')
    print('          f.write(json.dumps(example) + "\\n")')


if __name__ == "__main__":
    format_training_data_examples()
    data_quality_checks()
    common_data_quality_issues()
    data_synthesis_techniques()
    validation_split_strategy()
    data_preparation_checklist()
    save_training_data_example()

    print("\n" + "=" * 70)
    print("\nKey insight: Quality over quantity!")
    print("100 perfect examples > 1000 noisy examples")
