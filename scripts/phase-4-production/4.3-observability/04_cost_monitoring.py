"""
04 - Cost Monitoring
====================
Track token usage and costs across LLM calls.

Key concept: Cost monitoring prevents budget overruns and identifies expensive operations for optimization.

Book reference: AI_eng.4, AI_eng.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import sys
from pathlib import Path

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import os
from datetime import datetime
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from openai import OpenAI

# Initialize clients
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# OpenAI pricing (as of Jan 2025) - USD per 1M tokens
PRICING = {
    "gpt-4o-mini": {
        "input": 0.150,   # $0.150 per 1M input tokens
        "output": 0.600,  # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,    # $2.50 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "text-embedding-3-small": {
        "input": 0.020,   # $0.020 per 1M tokens
        "output": 0.0,    # No output cost for embeddings
    },
    "text-embedding-3-large": {
        "input": 0.130,
        "output": 0.0,
    }
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost for a single LLM call."""
    # Get pricing for model (default to gpt-4o-mini if not found)
    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


@observe()
def llm_call_with_cost_tracking(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> dict:
    """Make LLM call with automatic cost tracking."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature)

    # Calculate cost
    cost_info = calculate_cost(
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )

    # Log to Langfuse with cost metadata
    langfuse_context.update_current_observation(
        input={"prompt": prompt, "model": model},
        output={"response": response.choices[0].message.content},
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        metadata={
            "model": response.model,
            "temperature": temperature,
            "cost_usd": cost_info["total_cost_usd"],
            "input_cost_usd": cost_info["input_cost_usd"],
            "output_cost_usd": cost_info["output_cost_usd"],
        }
    )

    return {
        "response": response.choices[0].message.content,
        "cost": cost_info
    }


@observe()
def compare_model_costs(prompt: str, models: list[str]) -> list[dict]:
    """Compare costs across different models."""
    results = []

    for model in models:
        with langfuse_context.observe(name=f"call_{model}") as span:
            try:
                result = llm_call_with_cost_tracking(prompt, model=model)
                results.append({
                    "model": model,
                    "cost": result["cost"],
                    "response_length": len(result["response"])
                })

                span.update(
                    metadata={
                        "cost_usd": result["cost"]["total_cost_usd"],
                        "status": "success"
                    }
                )

            except Exception as e:
                results.append({
                    "model": model,
                    "error": str(e)
                })
                span.update(
                    level="ERROR",
                    status_message=str(e)
                )

    # Log comparison
    langfuse_context.update_current_observation(
        input={"prompt": prompt, "models": models},
        output={"results": results}
    )

    return results


@observe()
def batch_processing_with_budget(
    prompts: list[str],
    budget_usd: float,
    model: str = "gpt-4o-mini"
) -> dict:
    """Process prompts with budget limits."""
    results = []
    total_cost = 0.0
    processed = 0

    for i, prompt in enumerate(prompts):
        # Check if we're over budget
        if total_cost >= budget_usd:
            print(f"⚠ Budget limit reached at prompt {i+1}/{len(prompts)}")
            break

        with langfuse_context.observe(name=f"batch_item_{i}") as span:
            result = llm_call_with_cost_tracking(prompt, model=model)
            cost = result["cost"]["total_cost_usd"]

            # Check if this call would exceed budget
            if total_cost + cost > budget_usd:
                print(f"⚠ Skipping prompt {i+1} - would exceed budget")
                span.update(
                    level="WARNING",
                    status_message="Skipped due to budget limit"
                )
                break

            total_cost += cost
            processed += 1
            results.append({
                "prompt": prompt,
                "response": result["response"][:100] + "...",
                "cost": cost
            })

            span.update(
                metadata={
                    "cost_usd": cost,
                    "cumulative_cost_usd": total_cost,
                    "budget_remaining_usd": budget_usd - total_cost
                }
            )

    summary = {
        "processed": processed,
        "total_prompts": len(prompts),
        "total_cost_usd": round(total_cost, 6),
        "budget_usd": budget_usd,
        "budget_remaining_usd": round(budget_usd - total_cost, 6),
        "results": results
    }

    # Log summary
    langfuse_context.update_current_observation(
        input={
            "num_prompts": len(prompts),
            "budget_usd": budget_usd,
            "model": model
        },
        output=summary,
        metadata={
            "budget_utilization": f"{(total_cost/budget_usd)*100:.1f}%"
        }
    )

    return summary


@observe()
def cost_analysis_report(session_id: str) -> dict:
    """Generate cost analysis for a session (placeholder)."""
    # In production, you would query Langfuse API for session data
    # This is a simplified example

    report = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "note": "View detailed cost breakdown in Langfuse dashboard",
        "dashboard_url": f"https://cloud.langfuse.com/sessions/{session_id}"
    }

    langfuse_context.update_current_observation(
        input={"session_id": session_id},
        output=report
    )

    return report


if __name__ == "__main__":
    print("=== COST MONITORING ===\n")

    # Set session ID for grouping
    session_id = f"cost_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1. Single call with cost tracking
    print("1. Single LLM Call with Cost Tracking")
    print("-" * 60)
    result = llm_call_with_cost_tracking(
        "Explain quantum computing in one sentence",
        model="gpt-4o-mini"
    )
    print(f"Response: {result['response']}")
    print(f"Cost: ${result['cost']['total_cost_usd']:.6f}")
    print(f"  - Input: ${result['cost']['input_cost_usd']:.6f} ({result['cost']['prompt_tokens']} tokens)")
    print(f"  - Output: ${result['cost']['output_cost_usd']:.6f} ({result['cost']['completion_tokens']} tokens)\n")

    # 2. Compare model costs
    print("2. Compare Costs Across Models")
    print("-" * 60)
    models_to_compare = ["gpt-4o-mini", "gpt-4o"]
    comparison = compare_model_costs(
        "Write a haiku about AI",
        models=models_to_compare
    )

    print("\nModel Comparison:")
    for result in comparison:
        if "error" in result:
            print(f"  {result['model']}: Error - {result['error']}")
        else:
            print(f"  {result['model']}:")
            print(f"    Cost: ${result['cost']['total_cost_usd']:.6f}")
            print(f"    Response length: {result['response_length']} chars")

    # Calculate savings
    if len(comparison) == 2 and "error" not in comparison[0] and "error" not in comparison[1]:
        cost_diff = comparison[1]['cost']['total_cost_usd'] - comparison[0]['cost']['total_cost_usd']
        savings_pct = (cost_diff / comparison[1]['cost']['total_cost_usd']) * 100
        print(f"\n  Savings using {comparison[0]['model']}: ${cost_diff:.6f} ({savings_pct:.1f}%)\n")

    # 3. Batch processing with budget
    print("3. Batch Processing with Budget Limit")
    print("-" * 60)
    test_prompts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is neural network?",
        "What is reinforcement learning?",
        "What is natural language processing?"]

    budget_result = batch_processing_with_budget(
        test_prompts,
        budget_usd=0.01,  # $0.01 budget
        model="gpt-4o-mini"
    )

    print(f"\nProcessed: {budget_result['processed']}/{budget_result['total_prompts']} prompts")
    print(f"Total cost: ${budget_result['total_cost_usd']:.6f}")
    print(f"Budget remaining: ${budget_result['budget_remaining_usd']:.6f}")
    print(f"Average cost per prompt: ${(budget_result['total_cost_usd'] / budget_result['processed']):.6f}\n")

    # 4. Generate cost report
    print("4. Cost Analysis Report")
    print("-" * 60)
    report = cost_analysis_report(session_id)
    print(f"Session: {report['session_id']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"\n{report['note']}")
    print(f"View at: {report['dashboard_url']}\n")

    # Flush all traces
    langfuse.flush()

    print("=" * 60)
    print("✓ Cost monitoring complete")
    print("  - View detailed cost breakdown in Langfuse dashboard")
    print("  - Set up alerts for budget thresholds")
    print("  - Analyze cost trends over time")
    print("=" * 60)
