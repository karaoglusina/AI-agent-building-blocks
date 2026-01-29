"""
03 - Celery Setup
==================
Configure Celery with Redis for distributed task processing.

Key concept: Celery enables background job processing - offload long-running tasks (embeddings, fine-tuning, reports) to worker processes.

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
import time


def explain_celery_architecture():
    """Explain Celery architecture and components."""
    print("=== CELERY ARCHITECTURE ===\n")

    print("Components:\n")
    components = [
        "1. Client (Your App): Sends tasks to queue",
        "2. Message Broker (Redis): Stores tasks in queue",
        "3. Worker(s): Execute tasks from queue",
        "4. Result Backend (Redis): Stores task results"]

    for component in components:
        print(component)

    print("\n" + "=" * 70)
    print("\nFlow:")
    print("  Client → Broker (Redis) → Worker → Result Backend → Client")
    print("\n  1. Client calls task.delay() or task.apply_async()")
    print("  2. Task serialized and sent to Redis queue")
    print("  3. Worker picks up task from queue")
    print("  4. Worker executes task")
    print("  5. Result stored in Redis")
    print("  6. Client can check result with task_id")


def celery_configuration():
    """Show Celery configuration options."""
    print("\n" + "=" * 70)
    print("=== CELERY CONFIGURATION ===\n")

    print("Basic Celery app setup:\n")

    code = '''from celery import Celery

# Create Celery app
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',      # Message queue
    backend='redis://localhost:6379/1'      # Result storage
)

# Configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task time limits
    task_time_limit=300,        # 5 minutes hard limit
    task_soft_time_limit=240,   # 4 minutes soft limit

    # Concurrency
    worker_concurrency=4,       # Number of worker processes
    worker_prefetch_multiplier=1,  # Tasks per worker

    # Result expiration
    result_expires=3600,        # 1 hour
)
'''

    print(code)


def redis_setup_guide():
    """Guide for setting up Redis."""
    print("\n" + "=" * 70)
    print("=== REDIS SETUP ===\n")

    print("Option 1: Docker (Recommended)")
    print("-" * 40)
    docker_commands = '''# Start Redis container
docker run -d \\
  --name redis \\
  -p 6379:6379 \\
  redis:latest

# Check if running
docker ps

# View logs
docker logs redis

# Stop Redis
docker stop redis

# Remove container
docker rm redis
'''
    print(docker_commands)

    print("\nOption 2: Docker Compose")
    print("-" * 40)
    compose = '''# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:

# Start: docker-compose up -d
# Stop:  docker-compose down
'''
    print(compose)

    print("\nOption 3: Local Installation")
    print("-" * 40)
    install = '''# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis

# Verify installation
redis-cli ping  # Should return "PONG"
'''
    print(install)


def celery_task_examples():
    """Show example Celery task definitions."""
    print("\n" + "=" * 70)
    print("=== CELERY TASK EXAMPLES ===\n")

    print("Basic task:\n")

    basic = '''@app.task
def add(x, y):
    """Simple synchronous task."""
    return x + y

# Call the task
result = add.delay(4, 6)
print(f"Task ID: {result.id}")
print(f"Result: {result.get()}")  # Blocks until complete
'''
    print(basic)

    print("\nTask with retry:\n")

    retry_task = '''from celery import Task

@app.task(bind=True, max_retries=3)
def fetch_data(self, url):
    """Task with automatic retry on failure."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        # Retry after 60 seconds
        raise self.retry(exc=exc, countdown=60)
'''
    print(retry_task)

    print("\nLong-running task with progress:\n")

    progress_task = '''@app.task(bind=True)
def process_documents(self, doc_ids):
    """Task that reports progress."""
    total = len(doc_ids)
    for i, doc_id in enumerate(doc_ids):
        # Process document
        process_single_doc(doc_id)

        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': total}
        )

    return {'status': 'complete', 'processed': total}
'''
    print(progress_task)


def worker_commands():
    """Show Celery worker commands."""
    print("\n" + "=" * 70)
    print("=== CELERY WORKER COMMANDS ===\n")

    commands = '''# Start worker (basic)
celery -A tasks worker --loglevel=info

# Start worker with name
celery -A tasks worker --loglevel=info -n worker1@%h

# Start multiple workers
celery -A tasks worker --concurrency=4

# Start worker for specific queues
celery -A tasks worker -Q high_priority,default

# Autoscale workers (min=2, max=10)
celery -A tasks worker --autoscale=10,2

# Stop worker gracefully
celery -A tasks control shutdown

# Inspect active tasks
celery -A tasks inspect active

# Inspect registered tasks
celery -A tasks inspect registered

# Purge all tasks from queue
celery -A tasks purge
'''

    print(commands)


def task_routing():
    """Explain task routing and queues."""
    print("\n" + "=" * 70)
    print("=== TASK ROUTING & QUEUES ===\n")

    print("Route tasks to different queues:\n")

    routing = '''# Configure routes
app.conf.task_routes = {
    'tasks.generate_embeddings': {'queue': 'cpu_intensive'},
    'tasks.call_llm': {'queue': 'io_bound'},
    'tasks.send_email': {'queue': 'low_priority'},
}

# Send task to specific queue
result = task.apply_async(args=[...], queue='high_priority')

# Start workers for specific queues
# Worker 1: CPU-intensive tasks
celery -A tasks worker -Q cpu_intensive -n cpu_worker

# Worker 2: I/O-bound tasks
celery -A tasks worker -Q io_bound -n io_worker
'''
    print(routing)


def monitoring_tools():
    """Show monitoring and debugging tools."""
    print("\n" + "=" * 70)
    print("=== MONITORING & DEBUGGING ===\n")

    print("Flower - Web-based monitoring tool:\n")

    flower = '''# Install Flower
pip install flower

# Start Flower
celery -A tasks flower --port=5555

# Access dashboard
http://localhost:5555

Features:
- Real-time task monitoring
- Worker status and statistics
- Task history and results
- Rate limiting and control
'''
    print(flower)

    print("\nCelery Events:\n")

    events = '''# Start events monitoring
celery -A tasks events

# Capture events to console
celery -A tasks events --dump
'''
    print(events)


def production_considerations():
    """Production deployment considerations."""
    print("\n" + "=" * 70)
    print("=== PRODUCTION CONSIDERATIONS ===\n")

    considerations = [
        "1. Supervision: Use systemd, supervisord, or Docker to keep workers running",
        "2. Scaling: Multiple worker instances for high load",
        "3. Queue Separation: Different queues for different task types",
        "4. Monitoring: Use Flower or custom metrics (Prometheus)",
        "5. Error Handling: Implement retry logic and error notifications",
        "6. Rate Limiting: Prevent overwhelming external APIs",
        "7. Task Timeouts: Set time limits to prevent hanging tasks",
        "8. Result Backend: Clean up old results periodically",
        "9. Security: Use Redis password, encrypt sensitive data",
        "10. Backups: Redis persistence for task durability"
    ]

    for consideration in considerations:
        print(consideration)

    print("\n" + "=" * 70)
    print("\nSystemd service example:")

    systemd = '''# /etc/systemd/system/celery.service
[Unit]
Description=Celery Worker
After=network.target

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/app
Environment="PATH=/app/venv/bin"
ExecStart=/app/venv/bin/celery -A tasks worker \\
    --loglevel=info \\
    --pidfile=/var/run/celery/worker.pid
ExecStop=/app/venv/bin/celery -A tasks control shutdown
Restart=always

[Install]
WantedBy=multi-user.target
'''
    print(systemd)


def example_use_cases():
    """Show real-world use cases for Celery."""
    print("\n" + "=" * 70)
    print("=== REAL-WORLD USE CASES ===\n")

    use_cases = [
        "1. Batch Embeddings: Generate embeddings for 1000s of documents",
        "2. Fine-tuning: Train models without blocking web requests",
        "3. Report Generation: Create PDFs/exports asynchronously",
        "4. Email Campaigns: Send bulk emails in background",
        "5. Data Processing: ETL pipelines, data transformations",
        "6. Scheduled Tasks: Periodic cleanup, backups, sync",
        "7. LLM Batch Inference: Process queued LLM requests",
        "8. Vector DB Updates: Index new documents asynchronously",
        "9. Model Evaluation: Run eval suites in background",
        "10. Notifications: Send webhooks, Slack messages, etc."
    ]

    for use_case in use_cases:
        print(use_case)


def best_practices():
    """Show Celery best practices."""
    print("\n" + "=" * 70)
    print("=== BEST PRACTICES ===\n")

    practices = [
        "1. Keep Tasks Small: Break large jobs into smaller tasks",
        "2. Idempotent Tasks: Safe to retry without side effects",
        "3. Task Arguments: Pass IDs, not objects (serialization)",
        "4. Time Limits: Always set task timeouts",
        "5. Logging: Use structured logging for debugging",
        "6. Error Handling: Graceful failures with retries",
        "7. Queue Priority: Separate queues for different priorities",
        "8. Result Expiration: Clean up old results regularly",
        "9. Monitoring: Track task success/failure rates",
        "10. Testing: Write tests for task logic"
    ]

    for practice in practices:
        print(practice)


if __name__ == "__main__":
    explain_celery_architecture()
    celery_configuration()
    redis_setup_guide()
    celery_task_examples()
    worker_commands()
    task_routing()
    monitoring_tools()
    production_considerations()
    example_use_cases()
    best_practices()

    print("\n" + "=" * 70)
    print("\nKey insight: Celery = distributed task processing")
    print("Offload long-running AI tasks to background workers!")
    print("\nNext steps:")
    print("  1. Start Redis: docker run -d -p 6379:6379 redis")
    print("  2. Define tasks: See 04_background_task.py")
    print("  3. Start worker: celery -A tasks worker --loglevel=info")
    print("  4. Monitor tasks: See 05_task_status.py")
