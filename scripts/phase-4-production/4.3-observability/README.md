# Module 4.3: Observability

> *"Monitor, trace, and optimize AI applications in production"*

This module covers observability for AI systems using Langfuse, enabling tracing, cost monitoring, and custom metrics for production LLM applications.

## Files

| File | Topic | Key Concept |
|------|-------|-------------|
| `01_langfuse_setup.py` | Langfuse Setup | Initialize observability client and verify connection |
| `02_trace_llm_calls.py` | Trace LLM Calls | Log all LLM interactions with detailed metadata |
| `03_trace_rag_pipeline.py` | Trace RAG Pipeline | End-to-end RAG observability with nested spans |
| `04_cost_monitoring.py` | Cost Monitoring | Track token usage and costs across operations |
| `05_custom_metrics.py` | Custom Metrics | Add application-specific business metrics |

## Why Observability?

Observability is critical for production AI systems:
- **Debugging**: Trace issues through complex pipelines
- **Performance**: Identify bottlenecks and slow operations
- **Cost**: Monitor spending and prevent budget overruns
- **Quality**: Track response quality and user satisfaction
- **Optimization**: Data-driven improvements to prompts and models

## Core Concepts

### 1. Traces
End-to-end view of a request through your system:
```python
from langfuse.decorators import observe

@observe()
def my_llm_function(query: str) -> str:
    # Automatically traced
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

### 2. Spans
Individual steps within a trace (retrieval, generation, etc.):
```python
with langfuse_context.observe(name="retrieval"):
    docs = retrieve_documents(query)

with langfuse_context.observe(name="generation"):
    answer = generate_response(docs)
```

### 3. Scores
Evaluate quality, cost, latency, etc.:
```python
langfuse_context.score_current_trace(
    name="response_quality",
    value=quality_score,
    comment="User satisfaction rating"
)
```

### 4. Metadata
Additional context for filtering and analysis:
```python
langfuse_context.update_current_trace(
    user_id="user123",
    session_id="session456",
    tags=["production", "rag"],
    metadata={"cost_usd": 0.005}
)
```

## Typical Observability Stack

```
┌─────────────────────────────────────┐
│      Your AI Application            │
│  (FastAPI, Agents, RAG, etc.)       │
└──────────────┬──────────────────────┘
               │
               │ (traces, metrics)
               │
        ┌──────▼──────┐
        │  Langfuse   │
        │  Dashboard  │
        └─────────────┘
         - Traces
         - Metrics
         - Costs
         - Alerts
```

## Key Features

### Tracing
- **Automatic**: Use decorators for zero-config tracing
- **Manual**: Fine-grained control with SDK
- **Nested**: Track multi-step pipelines
- **Context**: User IDs, sessions, tags

### Cost Monitoring
- **Token tracking**: Log input/output tokens
- **Pricing**: Calculate costs per model
- **Budgets**: Set spending limits
- **Optimization**: Compare model costs

### Custom Metrics
- **Latency**: Response times
- **Quality**: Custom scoring
- **Business KPIs**: Domain-specific metrics
- **User feedback**: Satisfaction ratings

### Analytics
- **Dashboards**: Visualize trends
- **Filters**: By user, model, tag, etc.
- **Alerts**: Set threshold notifications
- **Exports**: Download data for analysis

## Prerequisites

### Install Langfuse
```bash
pip install langfuse
```

### Get API Keys
1. Sign up at [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project
3. Get your API keys from project settings
4. Set environment variables:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or self-hosted
```

### Verify Setup
```bash
python 01_langfuse_setup.py
```

## Running the Examples

### 1. Basic Setup
```bash
# Set up Langfuse and verify connection
python 01_langfuse_setup.py
```

### 2. Trace LLM Calls
```bash
# Log different types of LLM interactions
python 02_trace_llm_calls.py
```

### 3. Trace RAG Pipeline
```bash
# End-to-end RAG observability
python 03_trace_rag_pipeline.py
```

### 4. Monitor Costs
```bash
# Track spending across models
python 04_cost_monitoring.py
```

### 5. Custom Metrics
```bash
# Add business-specific metrics
python 05_custom_metrics.py
```

## Integration Patterns

### Decorator Pattern (Recommended)
```python
from langfuse.decorators import observe, langfuse_context

@observe()
def rag_pipeline(query: str) -> str:
    # Automatically traced
    docs = retrieve(query)
    answer = generate(docs)
    return answer
```

### Context Manager Pattern
```python
with langfuse_context.observe(name="custom_step") as span:
    result = do_something()
    span.update(output={"result": result})
```

### Manual SDK Pattern
```python
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(name="operation")
trace.update(output={"status": "success"})
langfuse.flush()
```

## Best Practices

### 1. Meaningful Names
```python
@observe(name="query_rewriting")  # Descriptive
def rewrite_query(query: str) -> str:
    ...
```

### 2. Rich Metadata
```python
langfuse_context.update_current_trace(
    user_id=user_id,
    session_id=session_id,
    tags=["production", "rag", "high_priority"],
    metadata={
        "customer_tier": "enterprise",
        "feature_flags": ["new_ranking"]
    }
)
```

### 3. Always Flush
```python
# At end of script or request
langfuse.flush()
```

### 4. Error Handling
```python
try:
    result = llm_call()
except Exception as e:
    langfuse_context.update_current_observation(
        level="ERROR",
        status_message=str(e)
    )
    raise
```

### 5. Cost Awareness
```python
# Log costs for budget tracking
langfuse_context.update_current_observation(
    metadata={
        "cost_usd": calculate_cost(tokens),
        "model": model_name
    }
)
```

## Langfuse Dashboard Features

### Traces View
- See all traces with timestamps
- Filter by user, session, tag
- Search by input/output
- Drill down into nested spans

### Analytics
- Token usage over time
- Cost breakdowns by model
- Latency percentiles
- Error rates

### Scores & Evaluation
- View quality scores
- Track user feedback
- A/B test comparisons
- Regression detection

### Datasets
- Create test sets from production data
- Run evaluations
- Compare model versions

## Production Considerations

### Performance
- **Async flushing**: Doesn't block your app
- **Batching**: Efficient network usage
- **Sampling**: Trace subset for high-volume apps

### Security
- **API keys**: Use environment variables
- **PII**: Avoid logging sensitive data
- **Self-hosted**: Deploy Langfuse on your infrastructure

### Monitoring
```python
# Set up alerts in Langfuse UI
- Cost exceeds $X per day
- Error rate > Y%
- Latency > Z seconds
- Quality score < threshold
```

### Retention
- Configure data retention policies
- Export historical data
- Archive old traces

## Common Use Cases

### 1. Debugging Production Issues
```python
# Find traces with errors
Filter: level = "ERROR"
Tag: "production"

# Trace shows:
- Input that caused error
- Full context and metadata
- Stack trace if logged
```

### 2. Cost Optimization
```python
# Compare model costs
- Run same prompts on different models
- Track costs by user/feature
- Identify expensive operations
- Set budget alerts
```

### 3. Quality Monitoring
```python
# Track response quality
- Log user feedback as scores
- Monitor quality trends
- Detect regressions
- A/B test prompt changes
```

### 4. Performance Optimization
```python
# Identify slow operations
- Sort traces by latency
- Find bottlenecks in pipeline
- Compare retrieval strategies
- Optimize based on data
```

## Advanced Features

### Prompt Management
```python
# Store and version prompts in Langfuse
prompt = langfuse.get_prompt("job_search_v2")
response = llm_call(prompt.compile(query=user_query))
```

### Datasets & Evaluation
```python
# Create dataset from production traces
dataset = langfuse.create_dataset("prod_queries")

# Run evaluations
for item in dataset.items:
    score = evaluate(item)
    langfuse.score(item.trace_id, value=score)
```

### Experiments
```python
# Track experiments
langfuse_context.update_current_trace(
    metadata={
        "experiment": "prompt_v3_test",
        "variant": "control"
    }
)
```

## OpenAI Integration

Langfuse has native OpenAI integration:
```python
from langfuse.openai import OpenAI

# Drop-in replacement
client = OpenAI()

# All calls automatically traced
response = client.chat.completions.create(...)
```

## Cost Reference

OpenAI pricing (as of Jan 2025):

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|--------------------:|---------------------:|
| gpt-4o-mini | $0.150 | $0.600 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4-turbo | $10.00 | $30.00 |
| text-embedding-3-small | $0.020 | - |
| text-embedding-3-large | $0.130 | - |

## Troubleshooting

### Connection Issues
```python
# Verify credentials
langfuse = Langfuse()
langfuse.trace(name="test")
langfuse.flush()  # Check for errors
```

### Missing Traces
```python
# Always flush at end
langfuse.flush()

# Or use context manager
from langfuse.decorators import langfuse_context
langfuse_context.flush()
```

### High Latency
```python
# Use async flushing (default)
langfuse = Langfuse(flush_at=10)  # Batch size

# Or disable for local testing
langfuse = Langfuse(enabled=False)
```

## Self-Hosted Option

Deploy Langfuse on your infrastructure:
```bash
docker run -d \
  -p 3000:3000 \
  -e DATABASE_URL=postgresql://... \
  langfuse/langfuse:latest
```

Set host URL:
```python
langfuse = Langfuse(
    host="http://localhost:3000"
)
```

## Book References

- `AI_eng.10` - Deployment and monitoring
- `AI_eng.4` - Cost optimization strategies

## Next Steps

After mastering observability:
- Module 4.4: Guardrails - Add safety checks to traced operations
- Module 4.5: Async & Background Jobs - Trace async workflows
- Module 4.7: Cloud Deployment - Deploy with observability enabled
- Module 5.1: Fine-tuning - Track fine-tuning experiments

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse Cloud](https://cloud.langfuse.com)
- [OpenAI Integration](https://langfuse.com/docs/integrations/openai)
- [Self-Hosting Guide](https://langfuse.com/docs/deployment/self-host)
