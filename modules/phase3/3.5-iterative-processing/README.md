# Module 3.5: Iterative Processing

> *"Process large content in stages"*

This module covers patterns for handling content that's too large for a single LLM call, using map-reduce, progressive summarization, refinement chains, and batch processing.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_map_reduce.py` | Map-Reduce Pattern | Process chunks independently then combine - handles arbitrarily large content |
| `02_progressive_summary.py` | Progressive Summarization | Summarize in stages (paragraph → section → document) for better quality |
| `03_refinement_chain.py` | Refinement Chain | Generate → Critique → Refine loop improves quality iteratively |
| `04_hierarchical_processing.py` | Hierarchical Processing | Process at multiple abstraction levels (detail → group → overview) |
| `05_batch_processing.py` | Large Batch Processing | Handle thousands of items with async + batching |

## Core Patterns

### 1. Map-Reduce
```
Large Doc → [Chunk 1, Chunk 2, ..., Chunk N]
         → Process each chunk (MAP)
         → Combine results (REDUCE)
```
Handles documents of any size by dividing work.

### 2. Progressive Summarization
```
Document → Paragraph summaries
         → Section summaries (combine paragraphs)
         → Document summary (combine sections)
```
Each level builds on the previous - better than direct summarization.

### 3. Refinement Chain
```
Initial output → Critique → Refined v2
              → Critique → Refined v3
              → ...
```
Iterative improvement converges to high quality.

### 4. Hierarchical Processing
```
Level 1: Process individual items (fine-grained)
Level 2: Summarize groups (mid-level)
Level 3: Create overview (high-level)
```
Works from detail to abstraction for better insights.

### 5. Batch Processing
```
[1000 items] → Batch 1 (50 items, async) → Results
            → Batch 2 (50 items, async) → Results
            → ...
```
Concurrent processing with rate limiting for throughput.

## When to Use Each Pattern

| Pattern | Best For | Complexity |
|---------|----------|------------|
| Map-Reduce | Very large documents (100k+ tokens) | Medium |
| Progressive Summary | Long documents needing quality | Medium |
| Refinement Chain | Quality-critical content | High |
| Hierarchical | Multi-level insights (detail → overview) | High |
| Batch | Processing 100s-1000s of items | Medium |

## Job Data Applications

These patterns excel at:
- **Map-Reduce**: Analyze trends across 10k+ job descriptions
- **Progressive Summary**: Create executive summary from 100+ jobs
- **Refinement Chain**: Polish job descriptions iteratively
- **Hierarchical**: Market analysis from jobs → roles → sectors
- **Batch**: Classify/extract data from thousands of jobs

## Performance Considerations

### Map-Reduce
- **Tokens**: N chunks × chunk_size + 1 combine call
- **Latency**: Serial = slow, Async = fast
- **Cost**: Linear with content size

### Progressive Summarization
- **Tokens**: Multiple passes, each smaller than last
- **Latency**: 3-5x single call
- **Cost**: 2-3x direct summarization, but higher quality

### Refinement Chain
- **Tokens**: 2-4x single generation
- **Latency**: 2-4 iterations
- **Cost**: High, but best quality

### Batch Processing
- **Throughput**: 5-10x with async
- **Rate Limits**: Respect TPM/RPM limits
- **Cost**: Same per item, faster overall

## Prerequisites

Install required libraries:

```bash
pip install openai asyncio
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Running the Scripts

Each script is self-contained:

```bash
python 01_map_reduce.py
python 02_progressive_summary.py
python 03_refinement_chain.py
python 04_hierarchical_processing.py
python 05_batch_processing.py
```

## Key Insights

1. **Divide and conquer works** - Map-reduce handles unbounded content
2. **Progressive > direct** - Multi-stage summarization beats single-shot
3. **Iteration improves quality** - Refinement chains converge to better output
4. **Abstraction layers help** - Hierarchical processing reveals patterns
5. **Async = speed** - Concurrent batch processing is 5-10x faster

## Pattern Combinations

These patterns often work together:

- **Map-Reduce + Hierarchical**: Process items, then group, then overview
- **Progressive + Refinement**: Summarize progressively, then refine
- **Batch + Map-Reduce**: Process thousands of large documents
- **Hierarchical + Progressive**: Multi-level summaries at each layer

## Book References

- `AI_eng.6` - RAG and large document handling
- `AI_eng.9` - Async processing and optimization
- `hands_on_LLM.II.7` - Chain of thought and multi-step reasoning
- `NLP_cook.9` - Text summarization techniques

## Next Steps

After mastering iterative processing:
- Module 3.6: FastAPI Basics
- Module 3.7: Clustering & Topics
- Module 3.8: Evaluation Systems
