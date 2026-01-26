"""
01 - Map-Reduce Pattern
========================
Process chunks then combine results.

Key concept: Split large tasks into independent chunks (map), process them, then combine (reduce).

Book reference: AI_eng.6, hands_on_LLM.II.7
"""

import asyncio
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = AsyncOpenAI()


def chunk_jobs(jobs: list[dict], chunk_size: int = 10) -> list[list[dict]]:
    """Split jobs into chunks for parallel processing."""
    return [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]


async def map_analyze_chunk(chunk: list[dict], chunk_id: int) -> dict:
    """Map: Analyze a chunk of jobs to extract key information."""
    job_summaries = []
    for job in chunk:
        job_summaries.append(f"- {job['title']}: {job['description'][:200]}")

    combined_text = "\n".join(job_summaries)

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Extract the most common required skills from these jobs. "
                           "List top 5 skills with approximate frequency."
            },
            {"role": "user", "content": combined_text}
        ]
    )

    return {
        "chunk_id": chunk_id,
        "job_count": len(chunk),
        "skills": response.output_text
    }


async def reduce_results(chunk_results: list[dict]) -> str:
    """Reduce: Combine chunk analyses into final summary."""
    combined_analyses = []
    total_jobs = 0

    for result in chunk_results:
        total_jobs += result["job_count"]
        combined_analyses.append(
            f"Chunk {result['chunk_id']} ({result['job_count']} jobs):\n{result['skills']}"
        )

    combined_text = "\n\n".join(combined_analyses)

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": f"Combine these skill analyses from {len(chunk_results)} chunks "
                           f"({total_jobs} total jobs). Create a unified top 10 skills list."
            },
            {"role": "user", "content": combined_text}
        ]
    )

    return response.output_text


async def map_reduce_analysis(jobs: list[dict], chunk_size: int = 10) -> str:
    """Run full map-reduce pipeline."""
    # Step 1: Split into chunks
    chunks = chunk_jobs(jobs, chunk_size)
    print(f"Split {len(jobs)} jobs into {len(chunks)} chunks")

    # Step 2: Map - Process chunks in parallel
    print("Map phase: Analyzing chunks in parallel...")
    map_tasks = [map_analyze_chunk(chunk, i) for i, chunk in enumerate(chunks)]
    chunk_results = await asyncio.gather(*map_tasks)
    print(f"  Processed {len(chunk_results)} chunks")

    # Step 3: Reduce - Combine results
    print("Reduce phase: Combining results...")
    final_result = await reduce_results(chunk_results)

    return final_result


async def main():
    """Demonstrate map-reduce pattern for job analysis."""
    print("=== MAP-REDUCE PATTERN ===\n")

    # Load a larger set of jobs
    jobs = load_sample_jobs(30)
    print(f"Analyzing {len(jobs)} jobs using map-reduce...\n")

    result = await map_reduce_analysis(jobs, chunk_size=10)

    print("\n=== FINAL RESULTS ===\n")
    print(result)
    print("\n" + "=" * 50)
    print("\nMap-reduce benefits:")
    print("  - Handles large datasets efficiently")
    print("  - Parallel processing reduces latency")
    print("  - Scalable to thousands of items")


if __name__ == "__main__":
    asyncio.run(main())
