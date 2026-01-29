# Module 2.6: Context Engineering

> *"Control what information goes into the LLM context"*

This module covers techniques for managing the limited context window effectively.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_context_window_basics.py` | Context Window Basics | Every model has a context limit - know your budget |
| `02_prompt_assembly.py` | Prompt Assembly | Modular prompts are easier to maintain and test |
| `03_context_prioritization.py` | Context Prioritization | Not all context is equal - prioritize by relevance |
| `04_context_compression.py` | Context Compression | Compression trades fidelity for capacity |
| `05_dynamic_system_prompts.py` | Dynamic System Prompts | System prompts can be templates filled at runtime |

## The Context Budget

```
┌─────────────────────────────────────────────┐
│              Context Window                  │
├─────────────────────────────────────────────┤
│ System Prompt          │ ~200 tokens        │
│ Retrieved Documents    │ ~2000 tokens       │
│ Conversation History   │ ~1000 tokens       │
│ User Query             │ ~100 tokens        │
│ Reserved for Response  │ ~500 tokens        │
├─────────────────────────────────────────────┤
│ Total Used             │ ~3800 tokens       │
│ Model Limit (4o-mini)  │ 128,000 tokens     │
└─────────────────────────────────────────────┘
```

## Token Efficiency Tips

1. Remove unnecessary whitespace
2. Use abbreviations in system prompts
3. Summarize old conversation turns
4. Use shorter field names in structured output
5. Compress retrieved documents

## Prerequisites

Install the required libraries:

```bash
pip install openai tiktoken
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_context_window_basics.py
python 02_prompt_assembly.py
# ... etc
```

## Context Engineering Strategies

| Strategy | When to Use | Trade-off |
|----------|-------------|-----------|
| Prioritization | Limited budget | May lose relevant info |
| Compression | Long conversations | Loses detail |
| Chunking | Large documents | Loses cross-chunk context |
| Dynamic prompts | Multi-task systems | Added complexity |

## Book References

- `AI_eng.5` - Prompt engineering
- `AI_eng.6` - RAG context management
- `hands_on_LLM.I.3` - Token economics
- `hands_on_LLM.II.7` - Conversation management
