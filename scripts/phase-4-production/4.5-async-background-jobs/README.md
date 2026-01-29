# Module 4.5: Async & Background Jobs

> *"Make your AI applications blazingly fast with concurrency"*

This module covers asynchronous programming and background job processing for AI applications - running multiple LLM calls in parallel and offloading long-running tasks to workers.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_asyncio_basics.py` | Asyncio Basics | Async/await fundamentals for concurrent I/O |
| `02_concurrent_llm.py` | Concurrent LLM Calls | Run multiple LLM API calls in parallel |
| `03_celery_setup.py` | Celery Setup | Configure Celery with Redis for task queue |
| `04_background_task.py` | Background Tasks | Run long tasks asynchronously with workers |
| `05_task_status.py` | Task Status Tracking | Monitor background job progress and results |

## Why Async & Background Jobs?

AI applications benefit dramatically from asynchronous processing:

### Performance Gains
- **5-10x faster**: Parallel LLM calls instead of sequential
- **Non-blocking**: API responds immediately while processing continues
- **Better throughput**: Handle more requests with same resources
- **Scalability**: Distribute work across multiple workers

### User Experience
- **Responsive**: No waiting for slow operations
- **Progress tracking**: Show users task status in real-time
- **Reliability**: Tasks survive crashes, can retry on failure

### Use Cases
- Batch embeddings generation (1000s of documents)
- Parallel LLM calls (sentiment + summary + keywords)
- Long-running tasks (fine-tuning, report generation)
- Scheduled jobs (cleanup, sync, backups)
- High-throughput APIs (queue requests, process async)

## Core Concepts

### 1. Async/Await (asyncio)

For **concurrent I/O operations** within a single process:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Sequential: 3 seconds
result1 = await call_llm(prompt1)
result2 = await call_llm(prompt2)
result3 = await call_llm(prompt3)

# Concurrent: 1 second (all at once!)
results = await asyncio.gather(
    call_llm(prompt1),
    call_llm(prompt2),
    call_llm(prompt3)
)
```

**When to use**: Multiple API calls, database queries, file I/O

### 2. Background Jobs (Celery)

For **distributed task processing** across multiple workers:

```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def generate_embeddings(doc_ids):
    """Process in background, may take minutes."""
    for doc_id in doc_ids:
        embedding = get_embedding(doc_id)
        save_to_db(embedding)
    return {"processed": len(doc_ids)}

# Client: Submit task and return immediately
result = generate_embeddings.delay([1, 2, 3, 4, 5])
return {"task_id": result.id}  # Return to user right away

# Later: Check status
status = AsyncResult(result.id).state  # PENDING/PROGRESS/SUCCESS
```

**When to use**: Long-running tasks, CPU-intensive work, scheduled jobs

## Prerequisites

### Install Dependencies

```bash
# Using uv (recommended)
uv pip install asyncio openai celery redis

# Using pip
pip install asyncio openai celery redis
```

### Start Redis (Message Broker)

**Option 1: Docker (Recommended)**
```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

**Option 2: Docker Compose**
```yaml
# docker-compose.yml
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
```

```bash
docker-compose up -d
```

**Option 3: Local Installation**
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# Verify
redis-cli ping  # Should return "PONG"
```

### Environment Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'
```

## Running the Examples

### 1. Asyncio Basics

```bash
python "scripts/phase-4-production/4.5-async-background-jobs/01_asyncio_basics.py"
```

Learn async/await fundamentals:
- `asyncio.gather()` - Run multiple coroutines concurrently
- `asyncio.create_task()` - Start task and continue working
- `asyncio.as_completed()` - Process results as they arrive
- `asyncio.wait_for()` - Set timeouts
- `asyncio.Semaphore()` - Rate limiting

### 2. Concurrent LLM Calls

```bash
python "scripts/phase-4-production/4.5-async-background-jobs/02_concurrent_llm.py"
```

Make multiple LLM calls in parallel:
- Parallel completions (5x faster)
- Batch embeddings (10-50x faster)
- Multi-model comparison
- Chain dependent calls
- Rate limiting with semaphore

### 3. Celery Setup

```bash
python "scripts/phase-4-production/4.5-async-background-jobs/03_celery_setup.py"
```

Learn Celery architecture and configuration:
- Components: client, broker, worker, backend
- Configuration options
- Worker commands
- Task routing and queues
- Monitoring tools (Flower)

### 4. Background Tasks

```bash
# Terminal 1: Start Celery worker
celery -A modules.phase4.4-5-async-background-jobs.04_background_task worker --loglevel=info

# Terminal 2: Run examples
python "scripts/phase-4-production/4.5-async-background-jobs/04_background_task.py"
```

Define and execute background tasks:
- Basic tasks
- AI-specific tasks (embeddings, sentiment analysis)
- Retry logic
- Task chains and groups
- Scheduled/periodic tasks

### 5. Task Status Tracking

```bash
python "scripts/phase-4-production/4.5-async-background-jobs/05_task_status.py"
```

Monitor task progress and results:
- Check task status
- Progress tracking
- API integration patterns
- Polling vs webhooks
- Error handling

## Architecture Patterns

### Pattern 1: Async API Endpoints

Fast API responses with concurrent operations:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /analyze
       ▼
┌─────────────────────────┐
│   FastAPI Endpoint      │
│   (async def)           │
└──────┬──────────────────┘
       │
       ├─────────┬─────────┬─────────┐
       ▼         ▼         ▼         ▼
    [LLM 1]  [LLM 2]  [LLM 3]  [LLM 4]

    All run concurrently with asyncio.gather()
    Total time = 1 LLM call (not 4x)
```

### Pattern 2: Background Job Queue

Offload long tasks to workers:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /embeddings
       ▼
┌─────────────────────────┐
│   FastAPI Endpoint      │
│   task.delay(...)       │
└──────┬──────────────────┘
       │ Return task_id immediately
       ▼
┌─────────────────────────┐
│   Redis (Message Queue) │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│   Celery Worker         │
│   (separate process)    │
│   - Generate embeddings │
│   - Store in vector DB  │
│   - Update status       │
└─────────────────────────┘
```

### Pattern 3: Hybrid (Async + Background)

Combine both for optimal performance:

```python
# API endpoint
@app.post("/batch-process")
async def batch_process(items: List[str]):
    # Quick async validation
    validated = await asyncio.gather(*[validate(item) for item in items])

    # Long processing in background
    result = process_batch.delay(validated)

    return {"task_id": result.id}
```

## Asyncio Patterns

### Pattern 1: Parallel LLM Calls

```python
async def analyze_text(text: str):
    """Run multiple analyses concurrently."""
    sentiment, summary, keywords = await asyncio.gather(
        get_sentiment(text),
        get_summary(text),
        extract_keywords(text)
    )
    return {
        "sentiment": sentiment,
        "summary": summary,
        "keywords": keywords
    }
```

### Pattern 2: Rate-Limited Calls

```python
async def batch_with_rate_limit(items: List[str]):
    """Process batch with max N concurrent calls."""
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)

    results = await asyncio.gather(*[process_with_limit(i) for i in items])
    return results
```

### Pattern 3: Timeout Handling

```python
async def call_with_timeout(prompt: str):
    """Set max wait time for LLM call."""
    try:
        result = await asyncio.wait_for(
            call_llm(prompt),
            timeout=10.0
        )
        return result
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
```

### Pattern 4: Error Handling

```python
async def parallel_with_error_handling(prompts: List[str]):
    """Continue on errors, collect successes."""
    results = await asyncio.gather(
        *[call_llm(p) for p in prompts],
        return_exceptions=True  # Don't stop on first error
    )

    # Separate successes from failures
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    return {"successes": successes, "failures": len(failures)}
```

## Celery Patterns

### Pattern 1: Basic Background Task

```python
@app.task
def generate_embeddings(text_ids: List[int]):
    """Generate embeddings in background."""
    embeddings = []
    for text_id in text_ids:
        text = fetch_text(text_id)
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return {"count": len(embeddings)}

# Usage
result = generate_embeddings.delay([1, 2, 3, 4, 5])
print(f"Task ID: {result.id}")
```

### Pattern 2: Progress Tracking

```python
@app.task(bind=True)
def process_documents(self, doc_ids: List[int]):
    """Track progress for long-running task."""
    total = len(doc_ids)

    for i, doc_id in enumerate(doc_ids):
        process_single_doc(doc_id)

        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': total,
                'percent': int((i + 1) / total * 100)
            }
        )

    return {"status": "complete", "processed": total}
```

### Pattern 3: Retry Logic

```python
@app.task(bind=True, max_retries=3)
def call_external_api(self, url: str):
    """Retry on failure with exponential backoff."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except RequestException as exc:
        # Retry: 2^retry_count seconds (2s, 4s, 8s)
        retry_count = self.request.retries
        countdown = 2 ** retry_count
        raise self.retry(exc=exc, countdown=countdown)
```

### Pattern 4: Task Chains

```python
from celery import chain

# Sequential: output of each feeds into next
workflow = chain(
    extract_text.s(pdf_id),
    generate_embedding.s(),
    store_in_db.s()
)

result = workflow.apply_async()
```

### Pattern 5: Task Groups

```python
from celery import group

# Parallel: all tasks run concurrently
job = group(
    process_doc.s(1),
    process_doc.s(2),
    process_doc.s(3),
    process_doc.s(4),
    process_doc.s(5)
)

result = job.apply_async()
results = result.get()  # Wait for all
```

### Pattern 6: Chords

```python
from celery import chord, group

# Group + callback when all complete
workflow = chord(
    group(
        process_doc.s(1),
        process_doc.s(2),
        process_doc.s(3)
    )
)(aggregate_results.s())  # Called with list of results

result = workflow.apply_async()
```

## FastAPI Integration

### Async Endpoints with Background Tasks

```python
from fastapi import FastAPI, BackgroundTasks
from celery.result import AsyncResult

app = FastAPI()

@app.post("/embeddings")
async def create_embeddings(texts: List[str]):
    """Start background embeddings task."""
    result = generate_embeddings_batch.delay(texts)
    return {
        "task_id": result.id,
        "status": "pending",
        "message": f"Processing {len(texts)} texts"
    }

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
        return {
            "task_id": task_id,
            "state": result.state,
            "progress": result.info
        }

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel running task."""
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True)
    return {"task_id": task_id, "status": "revoked"}
```

## Monitoring & Debugging

### Flower (Web UI)

```bash
# Install Flower
pip install flower

# Start Flower
celery -A tasks flower --port=5555

# Access dashboard
http://localhost:5555
```

**Features**:
- Real-time task monitoring
- Worker status and statistics
- Task history and results
- Rate limiting and control
- Manual task retry/revoke

### Celery Commands

```bash
# Inspect active tasks
celery -A tasks inspect active

# Inspect registered tasks
celery -A tasks inspect registered

# Inspect scheduled tasks
celery -A tasks inspect scheduled

# Purge all tasks from queue
celery -A tasks purge

# Stop worker gracefully
celery -A tasks control shutdown
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

@app.task(bind=True)
def process_data(self, data_id: int):
    logger.info(f"Starting task {self.request.id} for data {data_id}")

    try:
        result = expensive_operation(data_id)
        logger.info(f"Task {self.request.id} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}", exc_info=True)
        raise
```

## Performance Optimization

### Asyncio Optimization

1. **Use async libraries**: httpx (not requests), aiofiles, asyncpg
2. **Semaphore for rate limiting**: Control concurrent operations
3. **Connection pooling**: Reuse HTTP/DB connections
4. **Batch operations**: Group small operations
5. **Profile async code**: Use `asyncio` debugging tools

### Celery Optimization

1. **Worker concurrency**: Scale workers based on load
2. **Task routing**: Separate queues for different task types
3. **Prefetch settings**: Control tasks per worker
4. **Result expiration**: Clean up old results
5. **Serialization**: Use JSON (faster than pickle)

### Configuration Tuning

```python
app.conf.update(
    # Worker settings
    worker_concurrency=4,  # Number of worker processes
    worker_prefetch_multiplier=1,  # Tasks per worker
    worker_max_tasks_per_child=1000,  # Restart after N tasks

    # Task settings
    task_time_limit=300,  # 5 min hard limit
    task_soft_time_limit=240,  # 4 min soft limit
    task_acks_late=True,  # Ack after completion (reliability)
    task_reject_on_worker_lost=True,  # Requeue on worker crash

    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=True,  # Survive Redis restart

    # Connection settings
    broker_connection_retry_on_startup=True,
    broker_pool_limit=10,
)
```

## Production Deployment

### Docker Compose Setup

```yaml
version: '3.8'

services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  celery_worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  flower:
    build: .
    command: celery -A tasks flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis

volumes:
  redis_data:
```

### Scaling Workers

```bash
# Scale to 5 workers
docker-compose up -d --scale celery_worker=5

# Or manually
celery -A tasks worker -n worker1@%h &
celery -A tasks worker -n worker2@%h &
celery -A tasks worker -n worker3@%h &
```

### Systemd Service

```ini
# /etc/systemd/system/celery.service
[Unit]
Description=Celery Worker
After=network.target

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/app
Environment="PATH=/app/venv/bin"
ExecStart=/app/venv/bin/celery -A tasks worker \
    --loglevel=info \
    --pidfile=/var/run/celery/worker.pid
ExecStop=/app/venv/bin/celery -A tasks control shutdown
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable celery
sudo systemctl start celery

# Check status
sudo systemctl status celery
```

## Best Practices

### Asyncio Best Practices

1. **Async all the way**: Don't mix sync and async code
2. **Use gather() for parallel**: Don't await in loops
3. **Set timeouts**: Always use `wait_for()` for external calls
4. **Handle exceptions**: Use `return_exceptions=True`
5. **Rate limiting**: Use `Semaphore` to respect API limits
6. **Connection pooling**: Reuse clients and connections
7. **Profile performance**: Measure actual speedup
8. **Not for CPU**: Use multiprocessing for CPU-bound work

### Celery Best Practices

1. **Keep tasks small**: Break large jobs into smaller tasks
2. **Idempotent tasks**: Safe to retry without side effects
3. **Pass IDs, not objects**: Avoid large serialized data
4. **Set time limits**: Prevent hanging tasks
5. **Implement retries**: Handle transient failures
6. **Monitor tasks**: Track success/failure rates
7. **Clean up results**: Expire old task results
8. **Test thoroughly**: Unit test task logic
9. **Queue separation**: Different queues for priorities
10. **Graceful shutdown**: Handle termination signals

### General Best Practices

1. **Choose the right tool**:
   - Asyncio: Multiple API calls, concurrent I/O
   - Celery: Long-running tasks, distributed work
2. **Error handling**: Graceful degradation
3. **Logging**: Structured logs for debugging
4. **Monitoring**: Track performance metrics
5. **Testing**: Test async and background logic
6. **Documentation**: Document task behavior

## Common Pitfalls

### Asyncio Pitfalls

1. **Blocking operations**: Don't use `time.sleep()`, use `asyncio.sleep()`
2. **Sync libraries**: requests is blocking, use httpx
3. **Forgetting await**: Results in coroutine object, not result
4. **Too much concurrency**: Use semaphore for rate limiting
5. **No timeout**: External calls can hang forever

### Celery Pitfalls

1. **Large arguments**: Don't pass large objects, use IDs
2. **No idempotency**: Tasks fail on retry with side effects
3. **No time limits**: Tasks can run forever
4. **Forgetting result expiration**: Redis fills up
5. **Single queue**: No prioritization
6. **No monitoring**: Can't debug production issues
7. **Hard-coded config**: Use environment variables

## Use Cases

### AI Application Use Cases

1. **Batch Embeddings**: Generate embeddings for 1000s of documents
2. **Parallel LLM Calls**: Sentiment + summary + keywords
3. **Fine-tuning**: Train models without blocking API
4. **Report Generation**: Create PDFs/exports asynchronously
5. **Data Pipeline**: ETL for training data
6. **Model Evaluation**: Run eval suites in background
7. **Vector DB Updates**: Index new documents async
8. **Scheduled Jobs**: Periodic cleanup, sync, backups
9. **Email Campaigns**: Send bulk notifications
10. **Webhooks**: Notify external services

## Comparison: Asyncio vs Celery

| Feature | Asyncio | Celery |
|---------|---------|--------|
| **Use Case** | Concurrent I/O | Long-running tasks |
| **Execution** | Single process | Distributed workers |
| **Speedup** | 5-10x for I/O | Unlimited (scale workers) |
| **Setup** | Built-in Python | Requires Redis/broker |
| **State** | In-memory | Persistent (Redis) |
| **Monitoring** | Code-based | Flower UI |
| **Best For** | API calls, DB queries | Embeddings, fine-tuning |
| **Complexity** | Simple | More setup |

## Book References

- `AI_eng.9` - Async programming for AI applications
- `AI_eng.10` - Background job processing and task queues

## Next Steps

After mastering async and background jobs:
- Module 4.6: MCP Servers (Model Context Protocol)
- Module 4.7: Cloud Deployment (deploy async APIs)
- Module 4.3: Observability (monitor async tasks)
- Module 5.1: Fine-tuning (background training jobs)

## Additional Resources

- [Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Flower Monitoring](https://flower.readthedocs.io/)
- [Redis Documentation](https://redis.io/docs/)
- [FastAPI Async Guide](https://fastapi.tiangolo.com/async/)
