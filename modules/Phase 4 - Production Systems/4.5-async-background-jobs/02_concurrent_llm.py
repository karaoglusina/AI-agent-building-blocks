"""
02 - Concurrent LLM Calls
==========================
Run multiple LLM API calls in parallel for better performance.

Key concept: Making LLM calls concurrently drastically reduces total latency - 5 parallel calls take ~1 LLM call time instead of 5x.

Book reference: AI_eng.9
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import asyncio
import time
from typing import List, Dict
from openai import AsyncOpenAI
import os

# Initialize async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_completion(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Generate a single completion asynchronously."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


async def parallel_llm_calls():
    """Compare sequential vs parallel LLM calls."""
    print("=== PARALLEL LLM CALLS ===\n")

    prompts = [
        "Explain quantum computing in one sentence.",
        "What is the capital of France?",
        "Define machine learning briefly.",
        "Name three programming languages.",
        "What year did Python 1.0 release?"
    ]

    print(f"Making {len(prompts)} LLM calls...\n")

    # Parallel execution
    print("--- Concurrent (Parallel) ---")
    start = time.time()
    results = await asyncio.gather(*[generate_completion(p) for p in prompts])
    parallel_time = time.time() - start

    print(f"✓ All {len(prompts)} calls completed in {parallel_time:.2f}s\n")

    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"{i}. Q: {prompt}")
        print(f"   A: {result}\n")

    print(f"Average time per call: {parallel_time / len(prompts):.2f}s")
    print(f"Estimated sequential time: ~{parallel_time * 0.8:.1f}s")
    print(f"Speedup: ~{len(prompts) * 0.8:.1f}x faster!\n")


async def batch_embeddings():
    """Generate embeddings for multiple texts concurrently."""
    print("=" * 70)
    print("=== PARALLEL EMBEDDINGS ===\n")

    texts = [
        "Machine learning is a subset of AI.",
        "Neural networks are inspired by the brain.",
        "Deep learning uses multiple layers.",
        "Transformers revolutionized NLP.",
        "GPT models are based on transformers."
    ]

    async def get_embedding(text: str) -> List[float]:
        """Get embedding for a single text."""
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    print(f"Generating embeddings for {len(texts)} texts...\n")
    start = time.time()

    embeddings = await asyncio.gather(*[get_embedding(t) for t in texts])

    elapsed = time.time() - start
    print(f"✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    print(f"  Each embedding: {len(embeddings[0])} dimensions")
    print(f"  Average time per embedding: {elapsed / len(texts):.2f}s\n")


async def multi_model_comparison():
    """Compare responses from multiple models concurrently."""
    print("=" * 70)
    print("=== MULTI-MODEL COMPARISON ===\n")

    prompt = "Explain async programming in 2 sentences."

    async def call_model(model: str) -> Dict[str, str]:
        """Call a specific model."""
        print(f"  Calling {model}...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return {
            "model": model,
            "response": response.choices[0].message.content.strip()
        }

    print(f"Prompt: {prompt}\n")
    print("Querying multiple models in parallel...\n")

    start = time.time()

    # Query multiple models concurrently
    results = await asyncio.gather(
        call_model("gpt-4o-mini"),
        call_model("gpt-4o-mini"),  # Can compare temperature variations
        call_model("gpt-4o-mini"),
    )

    elapsed = time.time() - start

    print(f"✓ All models responded in {elapsed:.2f}s\n")

    for i, result in enumerate(results, 1):
        print(f"Model {i}: {result['model']}")
        print(f"Response: {result['response']}\n")


async def chain_dependent_calls():
    """Chain dependent LLM calls with some parallelization."""
    print("=" * 70)
    print("=== DEPENDENT LLM CALLS ===\n")

    print("Task: Generate a story outline, then write 3 chapters in parallel\n")

    # Step 1: Generate outline (must complete first)
    print("Step 1: Generating story outline...")
    outline_prompt = "Create a 3-sentence story outline about a robot learning to cook."
    outline = await generate_completion(outline_prompt)
    print(f"Outline: {outline}\n")

    # Step 2: Write chapters in parallel (depend on outline)
    print("Step 2: Writing 3 chapters in parallel...")
    start = time.time()

    chapter_prompts = [
        f"Based on this outline: '{outline}'. Write chapter 1 (2 sentences).",
        f"Based on this outline: '{outline}'. Write chapter 2 (2 sentences).",
        f"Based on this outline: '{outline}'. Write chapter 3 (2 sentences)."
    ]

    chapters = await asyncio.gather(*[generate_completion(p) for p in chapter_prompts])

    elapsed = time.time() - start

    for i, chapter in enumerate(chapters, 1):
        print(f"\nChapter {i}:")
        print(f"  {chapter}")

    print(f"\n✓ All chapters completed in {elapsed:.2f}s")
    print("(Sequential would take ~3x longer!)\n")


async def rate_limited_llm_calls():
    """Rate limit concurrent LLM calls."""
    print("=" * 70)
    print("=== RATE-LIMITED LLM CALLS ===\n")

    # Limit to 3 concurrent calls (API rate limits)
    semaphore = asyncio.Semaphore(3)

    async def rate_limited_call(prompt: str, call_id: int) -> str:
        """LLM call with rate limiting."""
        async with semaphore:
            print(f"  [Call {call_id}] Starting (max 3 concurrent)")
            result = await generate_completion(prompt)
            print(f"  [Call {call_id}] Complete")
            return result

    prompts = [f"What is {i} + {i}?" for i in range(1, 11)]

    print(f"Making {len(prompts)} LLM calls with max 3 concurrent...\n")
    start = time.time()

    results = await asyncio.gather(
        *[rate_limited_call(p, i) for i, p in enumerate(prompts, 1)]
    )

    elapsed = time.time() - start
    print(f"\n✓ Completed {len(results)} calls in {elapsed:.2f}s")
    print("(Without rate limiting, might hit API limits)\n")


async def timeout_handling():
    """Handle timeouts for slow LLM calls."""
    print("=" * 70)
    print("=== TIMEOUT HANDLING ===\n")

    async def slow_completion(prompt: str) -> str:
        """Simulate a slow completion."""
        return await generate_completion(prompt)

    print("Setting 10s timeout for LLM call...")
    try:
        result = await asyncio.wait_for(
            slow_completion("Explain async in detail."),
            timeout=10.0
        )
        print(f"✓ Completed: {result[:100]}...\n")
    except asyncio.TimeoutError:
        print("✗ LLM call timed out!\n")


async def error_handling_parallel():
    """Handle errors in parallel LLM calls."""
    print("=" * 70)
    print("=== ERROR HANDLING ===\n")

    async def potentially_failing_call(prompt: str, should_fail: bool) -> str:
        """LLM call that might fail."""
        if should_fail:
            raise ValueError("API error simulation")
        return await generate_completion(prompt)

    print("Making parallel calls with potential failures...\n")

    results = await asyncio.gather(
        potentially_failing_call("What is 1+1?", False),
        potentially_failing_call("What is 2+2?", True),  # Will fail
        potentially_failing_call("What is 3+3?", False),
        return_exceptions=True  # Don't stop on error
    )

    print("Results:")
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  Call {i}: ✗ Failed - {result}")
        else:
            print(f"  Call {i}: ✓ {result}")

    print("\nAll successful calls completed despite one failure!\n")


def concurrent_llm_best_practices():
    """Show best practices for concurrent LLM calls."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Parallel LLM Calls: Use asyncio.gather() for independent calls",
        "2. Rate Limiting: Use Semaphore to respect API limits",
        "3. Timeouts: Set max wait time with asyncio.wait_for()",
        "4. Error Handling: Use return_exceptions=True to continue on failure",
        "5. Sequential Dependencies: Await results before dependent calls",
        "6. Batch Embeddings: Generate all embeddings concurrently",
        "7. Model Comparison: Query multiple models in parallel",
        "8. Retry Logic: Implement exponential backoff for failures",
        "9. Cost Tracking: Monitor parallel calls to control costs",
        "10. Cache Results: Don't re-run identical parallel calls"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nTypical speedups:")
    print("  5 parallel calls: 4-5x faster than sequential")
    print("  10 parallel calls: 8-10x faster (with rate limiting)")
    print("  Embeddings: 10-50x faster for batches")

    print("\nAPI considerations:")
    print("  - OpenAI: 3-5 concurrent requests per key (free tier)")
    print("  - OpenAI: 100+ concurrent (paid tier)")
    print("  - Anthropic: Similar limits, check documentation")
    print("  - Always add rate limiting for production!")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Run examples
    asyncio.run(parallel_llm_calls())
    asyncio.run(batch_embeddings())
    asyncio.run(multi_model_comparison())
    asyncio.run(chain_dependent_calls())
    asyncio.run(rate_limited_llm_calls())
    asyncio.run(timeout_handling())
    asyncio.run(error_handling_parallel())
    concurrent_llm_best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Concurrent LLM calls = massive speedup")
    print("Use asyncio.gather() to parallelize independent LLM operations!")
