"""
05 - Task Status Tracking
==========================
Monitor background job progress and retrieve results.

Key concept: Track task status to show progress to users - essential for long-running operations like embeddings or fine-tuning.

Book reference: AI_eng.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from celery import Celery
from celery.result import AsyncResult
import time
from typing import Dict, Optional

# Configure Celery (same config as 04_background_task.py)
app = Celery(
    'background_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)


def check_task_status(task_id: str) -> Dict:
    """Check the status of a task."""
    result = AsyncResult(task_id, app=app)

    return {
        'task_id': task_id,
        'state': result.state,
        'ready': result.ready(),
        'successful': result.successful() if result.ready() else None,
        'failed': result.failed() if result.ready() else None,
        'result': result.result if result.ready() else None,
        'info': result.info  # Progress info or exception
    }


def monitor_task_until_complete(task_id: str, poll_interval: float = 1.0):
    """Monitor task until completion with progress updates."""
    print(f"Monitoring task: {task_id}\n")

    result = AsyncResult(task_id, app=app)

    while not result.ready():
        state = result.state

        if state == 'PENDING':
            print(f"Status: Waiting to start...")

        elif state == 'STARTED':
            print(f"Status: Task is running...")

        elif state == 'PROGRESS':
            # Custom progress state (from bind=True tasks)
            info = result.info
            if isinstance(info, dict):
                current = info.get('current', 0)
                total = info.get('total', 1)
                percent = info.get('percent', 0)
                print(f"Progress: {current}/{total} ({percent}%)")

        elif state == 'RETRY':
            print(f"Status: Retrying...")

        else:
            print(f"Status: {state}")

        time.sleep(poll_interval)

    # Task completed
    if result.successful():
        print(f"\n✓ Task completed successfully!")
        print(f"Result: {result.result}")
    else:
        print(f"\n✗ Task failed!")
        print(f"Error: {result.info}")

    return result.result


def get_task_result(task_id: str, timeout: Optional[float] = None) -> Dict:
    """Get task result (blocking until complete)."""
    result = AsyncResult(task_id, app=app)

    try:
        if timeout:
            output = result.get(timeout=timeout)
        else:
            output = result.get()

        return {
            'status': 'success',
            'result': output
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def revoke_task(task_id: str, terminate: bool = False):
    """Cancel a running task."""
    result = AsyncResult(task_id, app=app)

    if terminate:
        # Hard termination (may leave things in inconsistent state)
        result.revoke(terminate=True)
        print(f"Task {task_id} terminated (hard)")
    else:
        # Soft cancellation (task won't start if pending)
        result.revoke()
        print(f"Task {task_id} revoked (soft)")


def task_status_examples():
    """Show task status tracking examples."""
    print("=== TASK STATUS TRACKING ===\n")

    # Import tasks from previous script
    from modules.phase4.__background_jobs_04 import (
        add_numbers,
        generate_report,
        process_documents
    )

    # Example 1: Quick task
    print("1. Quick task with immediate check")
    result = add_numbers.delay(10, 20)
    task_id = result.id
    print(f"   Task ID: {task_id}")

    time.sleep(3)  # Wait for completion

    status = check_task_status(task_id)
    print(f"   State: {status['state']}")
    print(f"   Result: {status['result']}\n")

    # Example 2: Long task with monitoring
    print("2. Long task with progress monitoring")
    result = generate_report.delay(123)
    task_id = result.id
    print(f"   Task ID: {task_id}\n")

    # Monitor until complete
    monitor_task_until_complete(task_id, poll_interval=1.0)

    # Example 3: Task with progress updates
    print("\n3. Task with custom progress tracking")
    doc_ids = [1, 2, 3, 4, 5]
    result = process_documents.delay(doc_ids)
    task_id = result.id
    print(f"   Task ID: {task_id}\n")

    monitor_task_until_complete(task_id, poll_interval=0.5)


def api_integration_example():
    """Show how to integrate with API endpoints."""
    print("\n" + "=" * 70)
    print("=== API INTEGRATION PATTERN ===\n")

    print("FastAPI endpoint pattern:\n")

    code = '''from fastapi import FastAPI, BackgroundTasks
from celery.result import AsyncResult

app = FastAPI()

# Endpoint to start task
@app.post("/tasks/embeddings")
async def create_embeddings_task(texts: List[str]):
    """Start background embeddings task."""
    result = generate_embeddings_batch.delay(texts)
    return {
        "task_id": result.id,
        "status": "pending",
        "message": "Task submitted"
    }


# Endpoint to check task status
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Check task status."""
    result = AsyncResult(task_id, app=celery_app)

    if result.ready():
        if result.successful():
            return {
                "task_id": task_id,
                "state": "SUCCESS",
                "result": result.result
            }
        else:
            return {
                "task_id": task_id,
                "state": "FAILURE",
                "error": str(result.info)
            }
    else:
        # Task still running
        return {
            "task_id": task_id,
            "state": result.state,
            "progress": result.info if result.state == "PROGRESS" else None
        }


# Endpoint to cancel task
@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True)
    return {
        "task_id": task_id,
        "status": "revoked"
    }
'''

    print(code)


def task_states_explained():
    """Explain different task states."""
    print("\n" + "=" * 70)
    print("=== TASK STATES ===\n")

    states = [
        ("PENDING", "Task waiting to start (default state)"),
        ("STARTED", "Task has been started by a worker"),
        ("RETRY", "Task is being retried after failure"),
        ("PROGRESS", "Task is running and reporting progress (custom)"),
        ("SUCCESS", "Task completed successfully"),
        ("FAILURE", "Task failed with an exception"),
        ("REVOKED", "Task was cancelled before completion")]

    for state, description in states:
        print(f"{state:12} - {description}")


def polling_vs_webhooks():
    """Compare polling vs webhook patterns."""
    print("\n" + "=" * 70)
    print("=== POLLING VS WEBHOOKS ===\n")

    print("Polling Pattern (Simple)")
    print("-" * 40)
    polling = '''# Client repeatedly checks status
while True:
    response = requests.get(f"/tasks/{task_id}")
    if response.json()["state"] in ["SUCCESS", "FAILURE"]:
        break
    time.sleep(2)  # Poll every 2 seconds
'''
    print(polling)

    print("\nWebhook Pattern (Better)")
    print("-" * 40)
    webhook = '''# Server notifies client when task completes
@app.task(bind=True)
def process_with_webhook(self, data, callback_url):
    result = process_data(data)

    # Notify client
    requests.post(callback_url, json={
        "task_id": self.request.id,
        "status": "complete",
        "result": result
    })

    return result
'''
    print(webhook)

    print("\nWebSocket Pattern (Best for Real-time)")
    print("-" * 40)
    websocket = '''# Server pushes updates over WebSocket
@app.task(bind=True)
def process_with_websocket(self, data, connection_id):
    for i, item in enumerate(data):
        result = process_item(item)

        # Push update via WebSocket
        websocket_manager.send_to_client(connection_id, {
            "progress": (i + 1) / len(data),
            "current": i + 1,
            "total": len(data)
        })

    return {"status": "complete"}
'''
    print(websocket)


def task_result_lifecycle():
    """Explain task result lifecycle."""
    print("\n" + "=" * 70)
    print("=== TASK RESULT LIFECYCLE ===\n")

    print("Task result expiration:\n")

    lifecycle = '''# Configure result expiration
app.conf.result_expires = 3600  # Results expire after 1 hour

# Manually delete result
result = AsyncResult(task_id)
result.forget()  # Remove result from backend

# Periodic cleanup (in production)
@app.task
def cleanup_expired_results():
    """Remove old task results."""
    from celery.backends.redis import RedisBackend
    backend = app.backend
    # Clean up logic here
'''
    print(lifecycle)

    print("\nWhy expire results?")
    print("  - Prevents Redis/backend from filling up")
    print("  - Task results can be large (embeddings, reports)")
    print("  - Most results only needed temporarily")
    print("  - Set longer expiration for important tasks")


def error_handling_patterns():
    """Show error handling patterns."""
    print("\n" + "=" * 70)
    print("=== ERROR HANDLING ===\n")

    print("Handling task failures:\n")

    error_handling = '''def get_task_result_safe(task_id: str) -> Dict:
    """Safely get task result with error handling."""
    result = AsyncResult(task_id, app=app)

    try:
        if not result.ready():
            return {
                "status": "pending",
                "state": result.state,
                "progress": result.info
            }

        if result.successful():
            return {
                "status": "success",
                "result": result.result
            }
        else:
            # Task failed
            return {
                "status": "failure",
                "error": str(result.info),
                "traceback": result.traceback
            }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to get result: {str(e)}"
        }
'''
    print(error_handling)


def monitoring_dashboard_example():
    """Show monitoring dashboard concepts."""
    print("\n" + "=" * 70)
    print("=== MONITORING DASHBOARD ===\n")

    print("Track key metrics:\n")

    metrics = [
        "1. Active Tasks: Tasks currently running",
        "2. Pending Tasks: Tasks waiting in queue",
        "3. Completed Tasks: Successful completions (last hour/day)",
        "4. Failed Tasks: Failures with error details",
        "5. Average Duration: Time per task type",
        "6. Worker Health: Active workers and their status",
        "7. Queue Depth: Number of tasks per queue",
        "8. Success Rate: % of tasks completing successfully"
    ]

    for metric in metrics:
        print(metric)

    print("\n" + "=" * 70)
    print("\nTools for monitoring:")
    print("  - Flower: Web-based monitoring (celery -A tasks flower)")
    print("  - Prometheus + Grafana: Production metrics")
    print("  - Custom dashboard: Build with FastAPI + WebSocket")
    print("  - Sentry: Error tracking and alerting")


def best_practices_summary():
    """Summarize best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Store task IDs: Return immediately to client",
        "2. Poll Smartly: Use exponential backoff for polling",
        "3. Expire Results: Set appropriate expiration times",
        "4. Handle Errors: Graceful error messages to users",
        "5. Progress Updates: Use bind=True for long tasks",
        "6. Webhooks: Notify clients when tasks complete",
        "7. Idempotency: Safe to check status multiple times",
        "8. Task Metadata: Store user_id, created_at, etc.",
        "9. Result Cleanup: Periodically remove old results",
        "10. Monitoring: Track success/failure rates"
    ]

    for practice in practices:
        print(practice)


if __name__ == "__main__":
    print("Task Status Tracking Examples")
    print("=" * 70)
    print("\nThis script demonstrates task monitoring.")
    print("Make sure you have:")
    print("  1. Redis running: docker run -d -p 6379:6379 redis")
    print("  2. Celery worker: celery -A 04_background_task worker")
    print("\n" + "=" * 70)

    # Check if Redis is available
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("\n✓ Redis is running!")

        # Import check
        try:
            # Note: This import will fail if 04_background_task.py isn't structured properly
            # In production, tasks would be in a separate module
            print("\nTo run full examples:")
            print("  1. Start worker: celery -A 04_background_task worker")
            print("  2. In another terminal: python 05_task_status.py")
            print("\n" + "=" * 70)

            # Show conceptual examples
            task_states_explained()
            api_integration_example()
            polling_vs_webhooks()
            task_result_lifecycle()
            error_handling_patterns()
            monitoring_dashboard_example()
            best_practices_summary()

        except ImportError as e:
            print(f"\nNote: Task imports not available (expected): {e}")
            print("\nShowing conceptual examples instead...")

            task_states_explained()
            api_integration_example()
            polling_vs_webhooks()
            task_result_lifecycle()
            error_handling_patterns()
            monitoring_dashboard_example()
            best_practices_summary()

    except Exception as e:
        print(f"\n✗ Redis not available: {e}")
        print("\nStart Redis first:")
        print("  docker run -d -p 6379:6379 redis")

        print("\nShowing conceptual examples...\n")
        task_states_explained()
        api_integration_example()
        polling_vs_webhooks()
        task_result_lifecycle()
        error_handling_patterns()
        monitoring_dashboard_example()
        best_practices_summary()

    print("\n" + "=" * 70)
    print("\nKey insight: Task status tracking = great user experience")
    print("Show progress for long-running AI tasks!")
