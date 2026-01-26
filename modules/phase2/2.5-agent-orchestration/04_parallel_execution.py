"""
04 - Parallel Execution
=======================
Run multiple LLM calls concurrently.

Key concept: Async execution reduces latency when tasks are independent.

Book reference: AI_eng.9
"""

import asyncio
import time
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

client = AsyncOpenAI()


async def analyze_job(job: dict) -> dict:
    """Analyze a single job asynchronously."""
    response = await client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Extract the top 3 required skills from this job. Be brief."
            },
            {"role": "user", "content": job["description"][:500]}
        ]
    )
    return {
        "title": job["title"],
        "skills": response.output_text
    }


async def analyze_jobs_parallel(jobs: list[dict]) -> list[dict]:
    """Analyze multiple jobs in parallel."""
    tasks = [analyze_job(job) for job in jobs]
    results = await asyncio.gather(*tasks)
    return results


def analyze_jobs_sequential(jobs: list[dict]) -> list[dict]:
    """Analyze jobs sequentially (for comparison)."""
    from openai import OpenAI
    sync_client = OpenAI()
    
    results = []
    for job in jobs:
        response = sync_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "Extract the top 3 required skills. Be brief."},
                {"role": "user", "content": job["description"][:500]}
            ]
        )
        results.append({"title": job["title"], "skills": response.output_text})
    return results


async def main():
    """Compare parallel vs sequential execution."""
    jobs = load_sample_jobs(5)
    
    print("=== PARALLEL EXECUTION ===\n")
    print(f"Analyzing {len(jobs)} jobs...\n")
    
    # Parallel execution
    print("Running in PARALLEL...")
    start = time.time()
    parallel_results = await analyze_jobs_parallel(jobs)
    parallel_time = time.time() - start
    print(f"  Time: {parallel_time:.2f}s\n")
    
    # Sequential execution
    print("Running SEQUENTIALLY...")
    start = time.time()
    sequential_results = analyze_jobs_sequential(jobs)
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.2f}s\n")
    
    # Results
    print("=" * 50)
    print(f"Speedup: {sequential_time / parallel_time:.1f}x faster with parallel\n")
    
    print("=== RESULTS ===")
    for result in parallel_results:
        print(f"\n{result['title']}:")
        print(f"  {result['skills'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
