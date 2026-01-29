# Module 2.7: Memory Patterns

> *"Enable agents to remember across interactions"*

This module covers patterns for maintaining memory in conversational systems.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_conversation_buffer.py` | Conversation Buffer | Simple but grows unbounded - works for short chats |
| `02_sliding_window.py` | Sliding Window | Bounded memory - loses old context but stays in limits |
| `03_summary_memory.py` | Summary Memory | Compress history into summaries - keeps facts, loses details |
| `04_entity_memory.py` | Entity Memory | Extract and store key entities for persistent reference |
| `05_long_term_storage.py` | Long-Term Storage | Store facts that survive across sessions |
| `06_memory_retrieval.py` | Memory Retrieval | Store as embeddings, retrieve based on relevance |

## Memory Patterns Comparison

| Pattern | Capacity | Fidelity | Speed | Use Case |
|---------|----------|----------|-------|----------|
| Buffer | Unbounded | Perfect | Fast | Short conversations |
| Sliding Window | Fixed | Recent only | Fast | Long sessions |
| Summary | Compressed | Good | Medium | Long conversations |
| Entity | Key facts | Selected | Medium | Personalization |
| Long-term | Persistent | Selected | Medium | Cross-session |
| Retrieval | Large | Relevant | Slower | Knowledge-heavy |

## Job Data Application

- Remember user preferences ("I prefer remote jobs")
- Track skills mentioned across conversation
- Store job search history
- Recall relevant past interactions

## Prerequisites

Install the required libraries:

```bash
pip install openai tiktoken chromadb pydantic
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_conversation_buffer.py
python 02_sliding_window.py
# ... etc
```

## Memory Architecture

```
┌────────────────────────────────────────┐
│         Short-Term Memory              │
│  ┌─────────────┐  ┌─────────────┐     │
│  │   Buffer    │  │   Window    │     │
│  └─────────────┘  └─────────────┘     │
├────────────────────────────────────────┤
│         Long-Term Memory               │
│  ┌─────────────┐  ┌─────────────┐     │
│  │   Summary   │  │   Entities  │     │
│  └─────────────┘  └─────────────┘     │
│  ┌─────────────┐  ┌─────────────┐     │
│  │  Retrieval  │  │   Storage   │     │
│  └─────────────┘  └─────────────┘     │
└────────────────────────────────────────┘
```

## Book References

- `AI_eng.6` - Agent memory patterns
- `hands_on_LLM.II.7` - Conversation management
- `speach_lang.III.23` - Entity tracking and coreference
