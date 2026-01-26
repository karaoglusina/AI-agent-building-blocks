# Module 3.3: Advanced Memory

> *"Sophisticated memory management"*

This module covers advanced techniques for managing agent memory including structured storage, importance scoring, consolidation, preference tracking, and episodic recall.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_structured_memory.py` | Structured Memory | Use Pydantic models to enforce memory structure and validation |
| `02_memory_importance.py` | Memory Importance Scoring | Score memories by importance to decide what to keep and discard |
| `03_memory_consolidation.py` | Memory Consolidation | Combine redundant or related memories to reduce storage |
| `04_preference_memory.py` | Preference Memory System | Automatically detect, extract, and persist user preferences |
| `05_episodic_memory.py` | Episodic Memory | Store and retrieve specific episodes with context and timeline |

## Advanced Memory Patterns

### 1. Structured Memory
- Enforces type safety with Pydantic schemas
- Validates memory fields (confidence scores, dates, enums)
- Prevents malformed memories from entering the system
- Makes memories queryable and transformable

### 2. Importance Scoring
- Assigns scores to determine what to keep (0=trivial, 1=critical)
- Combines importance, recency, and access frequency
- Automatically prunes low-value memories
- Optimizes memory budget allocation

### 3. Memory Consolidation
- Merges redundant memories ("knows Python" + "Python expert" → "Python expert")
- Resolves contradictions by keeping newer information
- Reduces memory footprint over time
- Maintains source tracking for merged memories

### 4. Preference Memory
- Automatically detects preferences in conversation
- Stores in vector database for semantic retrieval
- Tracks preference strength (must-have vs. nice-to-have)
- Enables preference-aware recommendations

### 5. Episodic Memory
- Stores specific interaction episodes with context
- Maintains timeline of events
- Enables "remember when" queries
- Tracks sentiment and outcomes

## Memory Architecture

```
┌─────────────────────────────────────────────┐
│          Memory Management Layer             │
├─────────────────────────────────────────────┤
│  Importance    │  Consolidation  │ Structure │
│    Scoring     │      Engine     │ Validation│
└─────────────────────────────────────────────┘
         │                │              │
         ▼                ▼              ▼
┌─────────────────────────────────────────────┐
│            Memory Storage Layer              │
├─────────────────────────────────────────────┤
│   Preferences  │   Episodes   │   Facts     │
│   (ChromaDB)   │  (ChromaDB)  │  (Pydantic) │
└─────────────────────────────────────────────┘
         │                │              │
         ▼                ▼              ▼
┌─────────────────────────────────────────────┐
│              Retrieval Layer                 │
│  • Semantic search                          │
│  • Timeline queries                         │
│  • Category filtering                       │
│  • Importance thresholds                    │
└─────────────────────────────────────────────┘
```

## Job Search Application

These memory patterns enable sophisticated job search agents:

- **Structured Memory**: Store skills with proficiency levels and years of experience
- **Importance Scoring**: Keep critical career info (education, years exp) over trivial chat
- **Consolidation**: Merge "looking for Python jobs" + "want Python role" into one memory
- **Preference Memory**: Track all job preferences (remote, salary, company size, culture)
- **Episodic Memory**: Remember past interviews, applications, offers, and rejections

## Prerequisites

Install the required libraries:

```bash
pip install openai pydantic chromadb
```

Or using uv:

```bash
uv add openai pydantic chromadb
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_structured_memory.py
python 02_memory_importance.py
python 03_memory_consolidation.py
python 04_preference_memory.py
python 05_episodic_memory.py
```

## Memory Best Practices

### When to Use Each Pattern

| Pattern | Use When | Avoid When |
|---------|----------|------------|
| Structured | Need validation, have clear schema | Schema changes frequently |
| Importance | Limited memory budget | All memories equally important |
| Consolidation | Lots of redundant info | Need exact history preservation |
| Preference | Building personalized systems | No user-specific customization |
| Episodic | Need to recall specific events | Only need general facts |

### Performance Considerations

- **Structured Memory**: Fastest, no vector operations
- **Importance Scoring**: Fast, simple computation
- **Consolidation**: Slower, requires LLM calls for merging
- **Preference Memory**: Medium, uses embeddings
- **Episodic Memory**: Medium, uses embeddings

### Combining Patterns

Advanced systems often combine multiple patterns:

```python
# Example: Structured + Importance + Preference
memory = StructuredMemoryStore()
scorer = ImportanceScorer()
preferences = PreferenceMemory()

# Extract structured memories
memories = memory.extract_structured_memories(conversation)

# Score them
for mem in memories:
    mem.importance = scorer.score(mem)

# Store high-importance preferences
for mem in memories:
    if mem.importance > 0.7 and isinstance(mem, Preference):
        preferences.store(mem)
```

## Book References

- `AI_eng.6` - Agent memory architectures and patterns

## Next Steps

After mastering advanced memory:
- Module 3.4: Advanced RAG - Handle complex retrieval scenarios
- Module 3.5: Iterative Processing - Refine outputs through multiple passes
- Module 3.8: Evaluation Systems - Measure memory system performance
