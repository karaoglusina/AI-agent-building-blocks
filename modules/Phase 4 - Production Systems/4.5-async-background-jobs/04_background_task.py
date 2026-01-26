"""
04 - Background Tasks
=====================
Run long-running tasks asynchronously with Celery.

Key concept: Background tasks let your API respond immediately while processing continues - essential for user experience.

Book reference: AI_eng.10
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from celery import Celery
import time
from typing import List, Dict
import os

# Configure Celery
app = Celery(
    'background_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=240,  # 4 minutes warning
    result_expires=3600,  # Results expire after 1 hour
)


# ============================================================================
# Basic Background Tasks
# ============================================================================

@app.task
def add_numbers(x: int, y: int) -> int:
    """Simple background task."""
    time.sleep(2)  # Simulate work
    return x + y


@app.task
def generate_report(report_id: int) -> Dict:
    """Simulate long report generation."""
    print(f"[Task] Generating report {report_id}...")
    time.sleep(5)  # Simulate 5 seconds of work
    return {
        'report_id': report_id,
        'status': 'complete',
        'pages': 42,
        'generated_at': time.time()
    }


# ============================================================================
# AI-Specific Background Tasks
# ============================================================================

@app.task
def generate_embeddings_batch(texts: List[str]) -> Dict:
    """Generate embeddings for multiple texts."""
    print(f"[Task] Generating embeddings for {len(texts)} texts...")

    # Simulate embedding generation
    embeddings = []
    for i, text in enumerate(texts):
        time.sleep(0.5)  # Simulate API call
        # In production: embeddings.append(get_embedding(text))
        embeddings.append([0.1] * 1536)  # Mock embedding
        print(f"  Processed {i + 1}/{len(texts)}")

    return {
        'status': 'complete',
        'count': len(embeddings),
        'dimensions': 1536
    }


@app.task(bind=True)
def process_documents(self, doc_ids: List[int]) -> Dict:
    """Process multiple documents with progress tracking."""
    print(f"[Task] Processing {len(doc_ids)} documents...")

    total = len(doc_ids)
    results = []

    for i, doc_id in enumerate(doc_ids):
        # Simulate processing
        time.sleep(1)
        results.append({'doc_id': doc_id, 'status': 'processed'})

        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': total,
                'percent': int((i + 1) / total * 100)
            }
        )
        print(f"  Progress: {i + 1}/{total}")

    return {
        'status': 'complete',
        'processed': len(results),
        'results': results
    }


@app.task
def analyze_sentiment_batch(texts: List[str]) -> Dict:
    """Analyze sentiment for multiple texts."""
    print(f"[Task] Analyzing sentiment for {len(texts)} texts...")

    results = []
    for text in texts:
        time.sleep(0.5)  # Simulate LLM call
        # In production: sentiment = llm_analyze_sentiment(text)
        sentiment = "positive"  # Mock result
        results.append({'text': text[:50], 'sentiment': sentiment})

    return {
        'status': 'complete',
        'analyzed': len(results),
        'results': results
    }


# ============================================================================
# Tasks with Retry Logic
# ============================================================================

@app.task(bind=True, max_retries=3)
def fetch_external_data(self, url: str) -> Dict:
    """Fetch data with automatic retry on failure."""
    try:
        print(f"[Task] Fetching data from {url}...")
        time.sleep(1)

        # Simulate occasional failure
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise ConnectionError("Network error")

        return {'status': 'success', 'data': 'sample_data'}

    except ConnectionError as exc:
        print(f"[Task] Failed, retrying in 60s...")
        # Retry after 60 seconds
        raise self.retry(exc=exc, countdown=60)


@app.task(bind=True, max_retries=5)
def call_llm_with_retry(self, prompt: str) -> str:
    """Call LLM with exponential backoff retry."""
    try:
        print(f"[Task] Calling LLM with prompt: {prompt[:50]}...")
        time.sleep(2)

        # Simulate API rate limit error
        import random
        if random.random() < 0.2:  # 20% chance
            raise Exception("Rate limit exceeded")

        return f"Response to: {prompt}"

    except Exception as exc:
        # Exponential backoff: 2^retry_count seconds
        retry_count = self.request.retries
        countdown = 2 ** retry_count

        print(f"[Task] Retry {retry_count + 1}/5 in {countdown}s...")
        raise self.retry(exc=exc, countdown=countdown, max_retries=5)


# ============================================================================
# Scheduled/Periodic Tasks
# ============================================================================

@app.task
def cleanup_old_data():
    """Clean up old data - run periodically."""
    print("[Task] Running cleanup...")
    time.sleep(2)
    deleted_count = 42  # Simulate deletion
    print(f"[Task] Deleted {deleted_count} old records")
    return {'deleted': deleted_count}


@app.task
def sync_embeddings():
    """Sync embeddings to vector DB - run periodically."""
    print("[Task] Syncing embeddings to vector DB...")
    time.sleep(3)
    synced_count = 100
    print(f"[Task] Synced {synced_count} embeddings")
    return {'synced': synced_count}


# ============================================================================
# Task Chains and Groups
# ============================================================================

from celery import chain, group, chord


def chain_example():
    """Chain tasks - output of one becomes input to next."""
    print("=== TASK CHAIN EXAMPLE ===\n")

    # Define chain: add(2, 2) -> add(result, 3) -> add(result, 4)
    result = chain(
        add_numbers.s(2, 2),
        add_numbers.s(3),
        add_numbers.s(4)
    ).apply_async()

    print(f"Chain task ID: {result.id}")
    print("Waiting for chain to complete...")

    # Wait for result (blocking)
    final_result = result.get(timeout=10)
    print(f"Final result: {final_result}")  # Should be 11 (2+2+3+4)


def group_example():
    """Run multiple tasks in parallel."""
    print("\n" + "=" * 70)
    print("=== TASK GROUP EXAMPLE ===\n")

    # Run 5 tasks in parallel
    job = group(
        add_numbers.s(1, 1),
        add_numbers.s(2, 2),
        add_numbers.s(3, 3),
        add_numbers.s(4, 4),
        add_numbers.s(5, 5),
    )

    result = job.apply_async()
    print(f"Group task ID: {result.id}")
    print("Waiting for group to complete...")

    # Wait for all tasks
    results = result.get(timeout=10)
    print(f"Results: {results}")  # [2, 4, 6, 8, 10]


def chord_example():
    """Chord: group + callback when all complete."""
    print("\n" + "=" * 70)
    print("=== CHORD EXAMPLE ===\n")

    @app.task
    def sum_results(numbers):
        """Callback that sums all results."""
        return sum(numbers)

    # Run group, then callback
    result = chord(
        group(
            add_numbers.s(1, 1),
            add_numbers.s(2, 2),
            add_numbers.s(3, 3),
        )
    )(sum_results.s())

    print(f"Chord task ID: {result.id}")
    print("Waiting for chord to complete...")

    # Wait for final result
    final = result.get(timeout=10)
    print(f"Sum of results: {final}")  # 12 (2+4+6)


# ============================================================================
# Usage Examples
# ============================================================================

def basic_task_usage():
    """Show basic task usage patterns."""
    print("=== BASIC TASK USAGE ===\n")

    print("1. Fire and forget")
    result = add_numbers.delay(4, 6)
    print(f"   Task sent: {result.id}\n")

    print("2. Wait for result")
    result = add_numbers.delay(10, 20)
    print(f"   Task ID: {result.id}")
    print(f"   Result: {result.get(timeout=10)}\n")

    print("3. Check status without blocking")
    result = generate_report.delay(123)
    print(f"   Task ID: {result.id}")
    print(f"   State: {result.state}")
    print("   (Check 05_task_status.py for monitoring)\n")


def ai_task_examples():
    """Show AI-specific task examples."""
    print("=" * 70)
    print("=== AI TASK EXAMPLES ===\n")

    print("1. Batch embeddings")
    texts = [
        "Machine learning is a subset of AI.",
        "Neural networks are inspired by biology.",
        "Deep learning uses multiple layers.",
    ]
    result = generate_embeddings_batch.delay(texts)
    print(f"   Task ID: {result.id}")
    print(f"   Result: {result.get(timeout=10)}\n")

    print("2. Document processing with progress")
    doc_ids = [1, 2, 3, 4, 5]
    result = process_documents.delay(doc_ids)
    print(f"   Task ID: {result.id}")
    print("   (Monitor progress with 05_task_status.py)\n")

    print("3. Sentiment analysis batch")
    texts = ["Great product!", "Terrible service.", "It's okay."]
    result = analyze_sentiment_batch.delay(texts)
    print(f"   Task ID: {result.id}")
    print(f"   Result: {result.get(timeout=10)}\n")


def advanced_patterns():
    """Show advanced task patterns."""
    print("=" * 70)
    print("=== ADVANCED PATTERNS ===\n")

    print("1. Task with specific queue")
    result = generate_report.apply_async(
        args=[456],
        queue='reports'  # Route to specific queue
    )
    print(f"   Task ID: {result.id} (routed to 'reports' queue)\n")

    print("2. Task with priority")
    result = add_numbers.apply_async(
        args=[1, 2],
        priority=9  # Higher priority (0-9)
    )
    print(f"   Task ID: {result.id} (high priority)\n")

    print("3. Task with countdown (delay execution)")
    result = add_numbers.apply_async(
        args=[5, 5],
        countdown=5  # Start after 5 seconds
    )
    print(f"   Task ID: {result.id} (starts in 5s)\n")

    print("4. Task with ETA (scheduled)")
    from datetime import datetime, timedelta
    eta = datetime.utcnow() + timedelta(seconds=10)
    result = add_numbers.apply_async(
        args=[3, 7],
        eta=eta
    )
    print(f"   Task ID: {result.id} (scheduled for {eta})\n")


def best_practices_summary():
    """Summarize best practices."""
    print("=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Keep tasks small and focused",
        "2. Make tasks idempotent (safe to retry)",
        "3. Pass IDs, not objects (serialization)",
        "4. Use bind=True for progress tracking",
        "5. Set time limits to prevent hanging",
        "6. Implement retry logic for external calls",
        "7. Use queues for task prioritization",
        "8. Clean up results periodically",
        "9. Monitor task success/failure rates",
        "10. Test task logic independently"
    ]

    for practice in practices:
        print(practice)


if __name__ == "__main__":
    print("This file defines Celery tasks.")
    print("To run these examples, you need:")
    print("\n1. Start Redis:")
    print("   docker run -d -p 6379:6379 redis")
    print("\n2. Start Celery worker:")
    print("   celery -A 04_background_task worker --loglevel=info")
    print("\n3. Run this script to send tasks:")
    print("   python 04_background_task.py")
    print("\n" + "=" * 70)

    # Check if Redis is available
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("\n✓ Redis is running!")

        # Run examples
        print("\n" + "=" * 70)
        basic_task_usage()
        ai_task_examples()
        advanced_patterns()
        best_practices_summary()

    except Exception as e:
        print(f"\n✗ Redis not available: {e}")
        print("\nStart Redis first:")
        print("  docker run -d -p 6379:6379 redis")

    print("\n" + "=" * 70)
    print("\nKey insight: Background tasks = responsive applications")
    print("Offload long-running AI work to Celery workers!")
