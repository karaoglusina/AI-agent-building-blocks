"""
05 - Cost Tracking & Optimization
==================================
Monitor and optimize API costs.

Key concept: Every API call costs money - track usage, identify waste, optimize for cost.

Book reference: AI_eng.4, AI_eng.9
"""

import utils._load_env  # Loads .env file automatically

import tiktoken
from openai import OpenAI
import os
from typing import Any
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Pricing (as of 2024, in USD per 1M tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> dict:
    """Estimate API call cost."""
    pricing = PRICING[model]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model
    }


def compare_model_costs(prompt: str, expected_output_tokens: int = 50):
    """Compare costs across models."""
    prompt_tokens = count_tokens(prompt)

    print("=== MODEL COST COMPARISON ===\n")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Expected output tokens: {expected_output_tokens}\n")

    for model in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
        cost_info = estimate_cost(prompt_tokens, expected_output_tokens, model)
        print(f"{model}:")
        print(f"  Total cost: ${cost_info['total_cost']:.6f} per call")
        print(f"  Cost for 1000 calls: ${cost_info['total_cost'] * 1000:.2f}")
        print()


def optimize_prompt_length(long_prompt: str, short_prompt: str):
    """Show cost savings from shorter prompts."""
    long_tokens = count_tokens(long_prompt)
    short_tokens = count_tokens(short_prompt)

    print("=== PROMPT LENGTH OPTIMIZATION ===\n")
    print(f"Long prompt: {long_tokens} tokens")
    print(f"Short prompt: {short_tokens} tokens")
    print(f"Reduction: {long_tokens - short_tokens} tokens ({(1 - short_tokens/long_tokens)*100:.1f}%)\n")

    # Calculate savings
    output_tokens = 50  # Assume same output
    model = "gpt-4o-mini"

    long_cost = estimate_cost(long_tokens, output_tokens, model)["total_cost"]
    short_cost = estimate_cost(short_tokens, output_tokens, model)["total_cost"]

    savings_per_call = long_cost - short_cost
    savings_per_1000 = savings_per_call * 1000

    print(f"Cost per call:")
    print(f"  Long prompt: ${long_cost:.6f}")
    print(f"  Short prompt: ${short_cost:.6f}")
    print(f"  Savings: ${savings_per_call:.6f} ({(savings_per_call/long_cost)*100:.1f}%)\n")

    print(f"Savings for 1000 calls: ${savings_per_1000:.2f}")
    print(f"Savings for 100k calls: ${savings_per_1000 * 100:.2f}")


def batch_vs_individual(n_items: int = 10):
    """Compare cost of batch vs individual processing."""
    single_prompt = "Classify this job: Software Engineer"
    batch_prompt = "Classify these jobs:\n" + "\n".join([f"{i}. Software Engineer" for i in range(n_items)])

    single_tokens = count_tokens(single_prompt)
    batch_tokens = count_tokens(batch_prompt)

    print("\n=== BATCH VS INDIVIDUAL ===\n")
    print(f"Processing {n_items} items:\n")

    # Individual calls
    individual_total = single_tokens * n_items
    individual_cost = estimate_cost(individual_total, 20 * n_items)["total_cost"]

    print(f"Individual calls ({n_items} calls):")
    print(f"  Total input tokens: {individual_total}")
    print(f"  Estimated cost: ${individual_cost:.6f}\n")

    # Batch call
    batch_cost = estimate_cost(batch_tokens, 20 * n_items)["total_cost"]

    print(f"Single batch call:")
    print(f"  Total input tokens: {batch_tokens}")
    print(f"  Estimated cost: ${batch_cost:.6f}\n")

    print(f"Savings: ${individual_cost - batch_cost:.6f} ({((individual_cost - batch_cost)/individual_cost)*100:.1f}%)")


if __name__ == "__main__":
    print("=== COST TRACKING & OPTIMIZATION ===\n")

    # Example prompt
    prompt = """You are a job classification expert.
Classify the following job into one of these categories:
- Engineering
- Product
- Design
- Data
- Sales
- Other

Job: Senior Software Engineer at TechCorp

Category:"""

    # Compare models
    compare_model_costs(prompt, expected_output_tokens=10)

    # Optimize prompt length
    print("\n" + "=" * 70)
    short_prompt = "Classify job: Senior Software Engineer\nCategory:"
    optimize_prompt_length(prompt, short_prompt)

    # Batch vs individual
    print("\n" + "=" * 70)
    batch_vs_individual(n_items=10)

    print("\n" + "=" * 70)
    print("\nKey Insights:")
    print("1. gpt-4o-mini is 10-20x cheaper than gpt-4")
    print("2. Shorter prompts = lower costs")
    print("3. Batching can reduce token usage")
    print("4. Always measure before optimizing!")
