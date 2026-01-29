# Module 2.5: Agent Orchestration

> *"Connect capabilities into working agents"*

This module covers the patterns for building agents that can reason, use tools, and handle complex tasks.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_tool_calling_loop.py` | Tool Calling Loop | Agents are LLMs in a loop that call tools and observe results |
| `02_sequential_chain.py` | Sequential Chain | Break complex tasks into steps, passing output between them |
| `03_conditional_routing.py` | Conditional Routing | Use LLM to classify input, then branch to specialized handlers |
| `04_parallel_execution.py` | Parallel Execution | Async execution reduces latency for independent tasks |
| `05_tool_with_context.py` | Tools with Context | Maintain state across tool calls using a context object |
| `06_error_recovery.py` | Error Recovery | Build in retry logic, fallbacks, and graceful degradation |
| `07_agent_with_rag.py` | Agent with RAG | RAG as a tool lets agents decide when to retrieve |
| `08_multi_tool_agent.py` | Multi-Tool Agent | Give agents multiple tools - let LLM decide which to use |

## The Agent Loop

```
      ┌─────────────────────────────────────┐
      │                                     │
      ▼                                     │
   [THINK]  ──────►  [ACT]  ──────►  [OBSERVE]
   LLM decides      Execute          Get result
   what to do       tool call        from tool
```

## Job Data Application

- Agent that can search, filter, summarize, and compare jobs
- Handle follow-up questions with context
- Route queries to appropriate handlers
- Graceful error handling when services fail

## Prerequisites

Install the required libraries:

```bash
pip install openai pydantic chromadb
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_tool_calling_loop.py
python 02_sequential_chain.py
# ... etc
```

## Agent Patterns Summary

| Pattern | When to Use |
|---------|-------------|
| Sequential Chain | Steps depend on each other |
| Parallel Execution | Independent tasks |
| Conditional Routing | Different handlers for different inputs |
| Tool with Context | Need to maintain state |
| RAG Agent | Need to search knowledge |
| Multi-Tool | Multiple capabilities needed |

## Book References

- `AI_eng.6` - Agent architecture and failure modes
- `AI_eng.9` - Performance optimization (parallel)
- `hands_on_LLM.II.7` - Agent patterns
- `hands_on_LLM.II.8` - RAG integration
- `NLP_cook.10` - Conversational agents
