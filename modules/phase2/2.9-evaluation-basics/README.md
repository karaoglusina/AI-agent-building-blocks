# Module 2.9: Evaluation Basics

> *"Know if your system is working"*

This module covers essential evaluation techniques for AI systems.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_classification_metrics.py` | Classification Metrics | Different metrics for different use cases |
| `02_retrieval_metrics.py` | Retrieval Metrics | Measure how well you rank relevant items |
| `03_generation_metrics.py` | Generation Metrics | BLEU/ROUGE measure overlap with reference |
| `04_llm_as_judge.py` | LLM as Judge | LLMs can evaluate at scale |
| `05_simple_eval_loop.py` | Simple Eval Loop | Systematic evaluation finds issues early |
| `06_comparison_eval.py` | A/B Comparison | Comparative evaluation for decisions |
| `07_embedding_evaluation.py` | Embedding Evaluation | Test embeddings on your domain |

## Evaluation Strategy

```
┌─────────────────────────────────────────────┐
│           Evaluation Pipeline               │
├─────────────────────────────────────────────┤
│  1. Define success metrics                  │
│  2. Create test cases                       │
│  3. Run automated evaluation                │
│  4. Review failures                         │
│  5. Iterate on system                       │
│  6. Track metrics over time                 │
└─────────────────────────────────────────────┘
```

## Metric Selection Guide

| Task Type | Primary Metrics | Secondary |
|-----------|-----------------|-----------|
| Classification | Precision, Recall, F1 | Confusion Matrix |
| Retrieval | MRR, NDCG, Recall@K | MAP |
| Generation | ROUGE, LLM-judge | BLEU |
| Ranking | NDCG, MRR | Precision@K |

## Prerequisites

Install the required libraries:

```bash
pip install scikit-learn numpy rouge-score nltk sentence-transformers openai pydantic
python -c "import nltk; nltk.download('punkt')"
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_classification_metrics.py
python 02_retrieval_metrics.py
# ... etc
```

## Evaluation Best Practices

1. **Start with automated metrics** - Fast feedback loop
2. **Add LLM-as-judge** - More nuanced evaluation
3. **Include human eval** - Ground truth validation
4. **Track over time** - Catch regressions
5. **Test edge cases** - Where systems fail

## Book References

- `AI_eng.3` - Evaluation fundamentals
- `AI_eng.4` - Evaluation pipelines
- `hands_on_LLM.II.8` - Retrieval evaluation
- `hands_on_LLM.III.12` - LLM evaluation
- `speach_lang.I.4.7` - Classification metrics
- `speach_lang.II.13.6` - Generation metrics
