"""
05 - Large Batch Processing
============================
Handle thousands of items efficiently with batching and async.

Key concept: Process large datasets by batching requests and using concurrency - balance throughput with rate limits.

Book reference: AI_eng.9
"""

import asyncio
from openai import AsyncOpenAI
from typing import Any
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = AsyncOpenAI()


async def classify_job_async(job: dict[str, Any], semaphore: asyncio.Semaphore) -> dict:
    """Classify a single job asynchronously with rate limiting."""
    async with semaphore:  # Limit concurrent requests
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Classify job into one category: Engineering, Product, Design, Marketing, Sales, Operations, or Other"
                    },
                    {
                        "role": "user",
                        "content": f"Job title: {job['title']}\n\nCategory:"
                    }
                ],
                temperature=0.3,
                max_tokens=20
            )

            category = response.choices[0].message.content.strip()

            return {
                "id": job.get("id"),
                "title": job["title"],
                "category": category,
                "status": "success"
            }
        except Exception as e:
            return {
                "id": job.get("id"),
                "title": job["title"],
                "category": None,
                "status": "error",
                "error": str(e)
            }


async def process_batch(jobs: list[dict], batch_size: int = 10, max_concurrent: int = 5) -> list[dict]:
    """Process jobs in batches with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    # Process in batches
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} jobs)...")

        # Process batch concurrently
        tasks = [classify_job_async(job, semaphore) for job in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        print(f"Completed: {len(results)}/{len(jobs)}")

        # Small delay between batches to respect rate limits
        if i + batch_size < len(jobs):
            await asyncio.sleep(1)

    return results


def analyze_results(results: list[dict]) -> dict:
    """Analyze classification results."""
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    # Count categories
    category_counts = {}
    for result in successful:
        category = result["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "categories": category_counts,
        "success_rate": len(successful) / len(results) * 100
    }


async def batch_process_jobs(num_jobs: int = 50):
    """Main function to process large batch of jobs."""
    print(f"Loading {num_jobs} jobs...\n")
    jobs = load_sample_jobs(num_jobs)

    print(f"Processing {len(jobs)} jobs with async + batching...")
    print("Configuration:")
    print("  - Batch size: 10")
    print("  - Max concurrent: 5")
    print("  - Model: gpt-4o-mini\n")

    results = await process_batch(jobs, batch_size=10, max_concurrent=5)

    return results


if __name__ == "__main__":
    import time

    print("=== LARGE BATCH PROCESSING ===\n")

    start_time = time.time()

    # Run async batch processing
    results = asyncio.run(batch_process_jobs(num_jobs=50))

    elapsed = time.time() - start_time

    # Analyze results
    analysis = analyze_results(results)

    print("\n" + "=" * 70)
    print("=== RESULTS ===")
    print(f"Total jobs processed: {analysis['total']}")
    print(f"Successful: {analysis['successful']}")
    print(f"Failed: {analysis['failed']}")
    print(f"Success rate: {analysis['success_rate']:.1f}%")
    print(f"\nTime elapsed: {elapsed:.2f}s")
    print(f"Jobs per second: {analysis['total']/elapsed:.2f}")

    print("\n=== CATEGORY DISTRIBUTION ===")
    for category, count in sorted(analysis['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count}")

    print("\nBatch processing complete!")
    print("Key insight: Async + batching = efficient large-scale processing")
