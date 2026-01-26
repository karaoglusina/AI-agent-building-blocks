# Module 3.4: Advanced RAG

Advanced retrieval-augmented generation techniques for complex scenarios.

## Scripts

1. **01_query_rewriting.py** - Transform queries for better retrieval
2. **02_multi_hop_rag.py** - Multiple retrieval steps for complex questions
3. **03_self_rag.py** - Decide when retrieval is needed
4. **04_cross_encoder_rerank.py** - Rerank results with cross-encoder
5. **05_hybrid_search.py** - Combine BM25 + semantic search
6. **06_rag_fusion.py** - Reciprocal rank fusion from multiple queries

## Key Concepts

- Query rewriting improves retrieval by expanding and clarifying queries
- Multi-hop RAG handles questions requiring multiple information pieces
- Self-RAG agents decide when to retrieve vs. use parametric knowledge
- Cross-encoders provide more accurate relevance scoring
- Hybrid search combines keyword matching and semantic similarity
- RAG Fusion aggregates results from multiple query perspectives

## Usage

```bash
python 01_query_rewriting.py
```

## Prerequisites

- chromadb
- openai
- sentence-transformers (for reranking)
- rank-bm25 (for hybrid search)
