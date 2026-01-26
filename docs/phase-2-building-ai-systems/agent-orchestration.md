# Agent Orchestration

An agent is an LLM in a loop with tools. It thinks about what to do, takes an action, observes the result, and repeats until done. This is how you build systems that can actually do things - not just answer questions.

## Why This Matters

Some tasks require multiple steps. "Find remote ML jobs, compare the top 3, and tell me which has the best benefits" isn't a single retrieval. It's search, then comparison, then analysis. Agents can break down tasks and execute them step by step.

For our job market analyzer, agents enable queries like "Research the AI job market in Europe and create a summary report" - something that requires multiple searches, comparisons, and synthesis.

## The Key Ideas

### The Agent Loop

```
      ┌─────────────────────────────────────┐
      │                                     │
      ▼                                     │
   [THINK]  ──────►  [ACT]  ──────►  [OBSERVE]
   LLM decides      Execute          Get result
   what to do       tool call        from tool
```

1. **Think**: The LLM decides what action to take
2. **Act**: Execute the chosen tool
3. **Observe**: Feed the result back to the LLM
4. **Repeat** until the task is complete

### Tool Calling

OpenAI's function calling lets the model invoke tools:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search for job postings",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "location": {"type": "string"}
                }
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

# If the model wants to call a tool:
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the tool, feed result back
```

### The Basic Loop

```python
def run_agent(user_query: str):
    messages = [
        {"role": "system", "content": "You are a job market analyst with access to tools."},
        {"role": "user", "content": user_query}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        # If no tool call, we're done
        if not message.tool_calls:
            return message.content

        # Execute tools
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        messages.append(message)
```

### Error Recovery

Tools fail. Networks timeout. Handle it gracefully:

```python
def execute_tool_safely(tool_call, max_retries=2):
    for attempt in range(max_retries):
        try:
            return execute_tool(tool_call)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Tool failed: {e}"
            time.sleep(1)
```

Return errors as text. The LLM can decide what to do: retry, try a different approach, or tell the user.

### Multi-Tool Agents

Real agents have multiple capabilities:

```python
tools = [
    search_jobs_tool,
    get_job_details_tool,
    compare_jobs_tool,
    analyze_trends_tool,
    set_preference_tool
]
```

The LLM chooses which tool to use based on the task. Good tool descriptions help it choose correctly.

### Sequential vs Parallel

Some tool calls can run in parallel:

```python
# Sequential: wait for each result
result1 = search_jobs("ML engineer")
result2 = search_jobs("Data scientist")  # Waits

# Parallel: run concurrently
import asyncio
results = await asyncio.gather(
    search_jobs_async("ML engineer"),
    search_jobs_async("Data scientist")
)
```

Parallel execution reduces latency when tasks are independent.

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_tool_calling_loop.py | Basic agent loop with tools |
| 02_sequential_chain.py | Chain multiple steps |
| 03_conditional_routing.py | Branch based on LLM decisions |
| 04_parallel_execution.py | Run tool calls concurrently |
| 05_tool_with_context.py | Pass context between tool calls |
| 06_error_recovery.py | Handle tool failures |
| 07_agent_with_rag.py | RAG as an agent tool |
| 08_multi_tool_agent.py | Agent with multiple capabilities |

## Common Patterns

| Pattern | When to Use |
|---------|-------------|
| Sequential Chain | Steps depend on each other |
| Parallel Execution | Independent tasks |
| Conditional Routing | Different handlers for different inputs |
| RAG Tool | Need to search knowledge |
| Multi-Tool | Multiple capabilities needed |

## Things to Think About

- **How do you prevent infinite loops?** Set a maximum number of iterations. Detect when the agent is stuck.
- **What if the agent makes bad decisions?** Good tool descriptions and examples help. But sometimes you need guardrails.
- **When is an agent overkill?** For simple, predictable tasks, a hardcoded pipeline is simpler and more reliable. Agents shine when flexibility is needed.

## Related

- [RAG Pipeline](./rag-pipeline.md) - Often used as an agent tool
- [Classification & Routing](./classification-routing.md) - Route queries before agent processing
- [Memory Patterns](./memory-patterns.md) - Agents with memory
- [Advanced Agent Patterns](../phase-3-advanced-patterns/advanced-agent-patterns.md) - ReAct, planning, reflection

## Book References

- AI_eng.6 - Agent architecture
- hands_on_LLM.II.7 - Agent patterns
- NLP_cook.10 - Conversational agents
