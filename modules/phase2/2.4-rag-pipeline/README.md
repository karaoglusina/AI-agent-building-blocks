# Module 2.4: RAG Pipeline

> *"The most common LLM pattern: retrieve then generate"*

This module covers Retrieval-Augmented Generation (RAG) - the foundational pattern for grounding LLMs in your data.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_basic_rag.py` | Basic RAG Pipeline | RAG grounds responses in your data, reducing hallucination |
| `02_context_assembly.py` | Context Assembly | How you present context matters - structure affects quality |
| `03_source_citation.py` | Source Citation | Citations let users verify information and build trust |
| `04_no_results_handling.py` | Handling No Results | Always have a fallback - empty results shouldn't break UX |
| `05_rag_with_filters.py` | RAG with Filters | Filters narrow results before semantic ranking |
| `06_rag_evaluation.py` | RAG Evaluation | Evaluate retrieval separately from generation |

## The RAG Pattern

```
Query → Retrieve → Augment → Generate → Answer
         ↓           ↓          ↓
      Vector DB   Prompt    LLM Call
```

## Job Data Application

- Q&A over job database: "What Python jobs are in Amsterdam?"
- Compare job requirements across roles
- Summarize trends in job postings
- Find similar jobs based on description

## Prerequisites

Install the required libraries:

```bash
pip install openai chromadb numpy
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_basic_rag.py
python 02_context_assembly.py
# ... etc
```

## Evaluation Metrics Reference

| Metric | Question Answered |
|--------|-------------------|
| Precision@K | What fraction of top-K are relevant? |
| Recall@K | What fraction of relevant docs in top-K? |
| MRR | How early is the first relevant result? |
| NDCG@K | How good is the overall ranking? |

## Book References

- `AI_eng.6` - RAG architecture and patterns
- `AI_eng.3` - Evaluation methods
- `hands_on_LLM.II.8` - Retrieval systems
- `speach_lang.II.14` - Question answering
