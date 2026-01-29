"""
01 - Asyncio Basics
===================
Async/await fundamentals for concurrent Python execution.

Key concept: Async/await enables concurrent I/O operations without blocking - crucial for AI applications making multiple API calls.

Book reference: AI_eng.9
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

import asyncio
import time
from typing import List


async def fetch_data(name: str, delay: float) -> str:
    """Simulate async data fetching (e.g., API call)."""
    print(f"  [{name}] Starting fetch... (will take {delay}s)")
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"  [{name}] Fetch complete!")
    return f"Data from {name}"


def sync_fetch_data(name: str, delay: float) -> str:
    """Synchronous version for comparison."""
    print(f"  [{name}] Starting fetch... (will take {delay}s)")
    time.sleep(delay)  # Blocking sleep
    print(f"  [{name}] Fetch complete!")
    return f"Data from {name}"


async def async_vs_sync_comparison():
    """Compare async vs sync execution times."""
    print("=== ASYNC VS SYNC COMPARISON ===\n")

    # Synchronous execution (blocking)
    print("--- Synchronous (Sequential) ---")
    start = time.time()
    result1 = sync_fetch_data("API-1", 1.0)
    result2 = sync_fetch_data("API-2", 1.0)
    result3 = sync_fetch_data("API-3", 1.0)
    sync_time = time.time() - start
    print(f"Total time: {sync_time:.2f}s\n")

    # Asynchronous execution (concurrent)
    print("--- Asynchronous (Concurrent) ---")
    start = time.time()
    results = await asyncio.gather(
        fetch_data("API-1", 1.0),
        fetch_data("API-2", 1.0),
        fetch_data("API-3", 1.0)
    )
    async_time = time.time() - start
    print(f"Total time: {async_time:.2f}s\n")

    print(f"Speedup: {sync_time / async_time:.1f}x faster with async!")


async def asyncio_gather_example():
    """Show asyncio.gather for running multiple tasks."""
    print("\n" + "=" * 70)
    print("=== ASYNCIO.GATHER ===\n")

    print("Running 5 tasks concurrently...")
    start = time.time()

    # All tasks run concurrently
    results = await asyncio.gather(
        fetch_data("Task-1", 0.5),
        fetch_data("Task-2", 1.0),
        fetch_data("Task-3", 0.3),
        fetch_data("Task-4", 0.7),
        fetch_data("Task-5", 0.9))

    elapsed = time.time() - start
    print(f"\nAll tasks completed in {elapsed:.2f}s")
    print(f"Results: {results}")


async def asyncio_create_task_example():
    """Show asyncio.create_task for managing tasks individually."""
    print("\n" + "=" * 70)
    print("=== ASYNCIO.CREATE_TASK ===\n")

    print("Creating tasks and managing them individually...")

    # Create tasks
    task1 = asyncio.create_task(fetch_data("Task-A", 1.0))
    task2 = asyncio.create_task(fetch_data("Task-B", 0.5))

    # Do other work while tasks run in background
    print("  [Main] Doing other work while tasks run...")
    await asyncio.sleep(0.2)
    print("  [Main] Still doing work...")

    # Wait for specific tasks
    result1 = await task1
    result2 = await task2

    print(f"\nResults: {result1}, {result2}")


async def error_handling_with_gather():
    """Show error handling in async operations."""
    print("\n" + "=" * 70)
    print("=== ERROR HANDLING ===\n")

    async def task_that_fails(name: str):
        """Task that raises an exception."""
        await asyncio.sleep(0.5)
        if name == "Task-2":
            raise ValueError(f"{name} failed!")
        return f"{name} succeeded"

    print("--- With return_exceptions=True ---")
    results = await asyncio.gather(
        task_that_fails("Task-1"),
        task_that_fails("Task-2"),  # This will fail
        task_that_fails("Task-3"),
        return_exceptions=True  # Don't stop on first exception
    )

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  Task-{i}: Failed - {result}")
        else:
            print(f"  Task-{i}: {result}")

    print("\n--- With return_exceptions=False (default) ---")
    try:
        results = await asyncio.gather(
            task_that_fails("Task-1"),
            task_that_fails("Task-2"),  # This will fail
            task_that_fails("Task-3"),
            return_exceptions=False
        )
    except ValueError as e:
        print(f"  Execution stopped on first error: {e}")


async def asyncio_as_completed_example():
    """Process results as they complete."""
    print("\n" + "=" * 70)
    print("=== ASYNCIO.AS_COMPLETED ===\n")

    print("Processing results as they arrive...\n")

    tasks = [
        fetch_data("Quick", 0.3),
        fetch_data("Medium", 1.0),
        fetch_data("Slow", 1.5),
        fetch_data("Fast", 0.5)]

    # Process results in completion order
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"  Got result: {result}")


async def timeout_example():
    """Show timeout handling."""
    print("\n" + "=" * 70)
    print("=== TIMEOUT HANDLING ===\n")

    async def slow_operation():
        print("  Starting slow operation...")
        await asyncio.sleep(5)
        return "Operation complete"

    print("--- Operation with 2 second timeout ---")
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
        print(f"  Result: {result}")
    except asyncio.TimeoutError:
        print("  ✗ Operation timed out!")

    print("\n--- Operation with 10 second timeout ---")
    try:
        # This would succeed but we'll use a faster version for demo
        async def fast_operation():
            await asyncio.sleep(0.5)
            return "Quick result"

        result = await asyncio.wait_for(fast_operation(), timeout=10.0)
        print(f"  ✓ Result: {result}")
    except asyncio.TimeoutError:
        print("  ✗ Operation timed out!")


async def semaphore_example():
    """Limit concurrent operations with semaphore."""
    print("\n" + "=" * 70)
    print("=== SEMAPHORE (RATE LIMITING) ===\n")

    # Only allow 2 concurrent operations
    semaphore = asyncio.Semaphore(2)

    async def rate_limited_fetch(name: str, delay: float):
        async with semaphore:
            print(f"  [{name}] Acquired semaphore (max 2 concurrent)")
            await asyncio.sleep(delay)
            print(f"  [{name}] Released semaphore")
            return f"Data from {name}"

    print("Running 5 tasks with max 2 concurrent...\n")
    start = time.time()

    results = await asyncio.gather(
        rate_limited_fetch("Task-1", 1.0),
        rate_limited_fetch("Task-2", 1.0),
        rate_limited_fetch("Task-3", 1.0),
        rate_limited_fetch("Task-4", 1.0),
        rate_limited_fetch("Task-5", 1.0))

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s")
    print("(Without semaphore, would take ~1s. With semaphore(2), takes ~3s)")


def asyncio_best_practices():
    """Show asyncio best practices."""
    print("\n" + "=" * 70)
    print("=== ASYNCIO BEST PRACTICES ===\n")

    practices = [
        "1. Use async/await: Make I/O operations non-blocking",
        "2. asyncio.gather(): Run multiple tasks concurrently",
        "3. asyncio.create_task(): Start task and continue working",
        "4. asyncio.as_completed(): Process results as they arrive",
        "5. asyncio.wait_for(): Set timeouts for operations",
        "6. asyncio.Semaphore(): Limit concurrent operations (rate limiting)",
        "7. return_exceptions=True: Handle errors gracefully",
        "8. Don't mix sync/async: Use async all the way down",
        "9. Use async libraries: httpx (not requests), aiofiles, etc.",
        "10. Profile: Use async where you have I/O, not CPU-bound work"
    ]

    for practice in practices:
        print(practice)

    print("\n" + "=" * 70)
    print("\nWhen to use async:")
    print("  ✓ Multiple API calls (LLM, embeddings, external APIs)")
    print("  ✓ Database queries (with async drivers)")
    print("  ✓ File I/O (with aiofiles)")
    print("  ✓ Network operations (HTTP, WebSocket)")
    print("  ✗ CPU-bound work (use multiprocessing instead)")
    print("  ✗ Single synchronous operation (overhead not worth it)")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(async_vs_sync_comparison())
    asyncio.run(asyncio_gather_example())
    asyncio.run(asyncio_create_task_example())
    asyncio.run(error_handling_with_gather())
    asyncio.run(asyncio_as_completed_example())
    asyncio.run(timeout_example())
    asyncio.run(semaphore_example())
    asyncio_best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Async = concurrent I/O = faster AI applications")
    print("Use asyncio.gather() for parallel LLM calls, embeddings, and API requests!")
