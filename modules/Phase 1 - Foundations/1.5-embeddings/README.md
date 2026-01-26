# Lesson 05: Embeddings

## Overview
Embeddings are vector representations of text that capture semantic meaning. Similar texts have similar vectors. This is the foundation for semantic search, clustering, and RAG.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_basic_embedding.py` | Create a single embedding | `python 01_basic_embedding.py` |
| `02_embedding_models.py` | Compare embedding models | `python 02_embedding_models.py` |
| `03_batch_embeddings.py` | Embed multiple texts efficiently | `python 03_batch_embeddings.py` |
| `04_cosine_similarity.py` | Measure vector similarity | `python 04_cosine_similarity.py` |
| `05_simple_search.py` | Basic vector search | `python 05_simple_search.py` |
| `06_job_search.py` | Search your job data | `python 06_job_search.py` |
| `07_save_load_embeddings.py` | Persist embeddings to disk | `python 07_save_load_embeddings.py` |
| `08_numpy_operations.py` | Essential numpy for vectors | `python 08_numpy_operations.py` |

## Key Takeaways

1. **Embeddings** = text → vector of floats (typically 1536 or 3072 dims)
2. **Cosine similarity** = measure how similar two vectors are (0-1 scale)
3. **text-embedding-3-small** = best for most use cases
4. **Batch your requests** = embed multiple texts in one API call
5. **Cache embeddings** = compute once, reuse forever

## The Math

```
Text → Embedding Model → [0.023, -0.041, 0.112, ...] (1536 numbers)
```

Similar texts → similar vectors → high cosine similarity

## Vector Search Algorithm

```python
# 1. Pre-compute: Embed all documents
doc_embeddings = [embed(doc) for doc in documents]

# 2. At query time: Embed the query
query_embedding = embed(query)

# 3. Compare: Calculate similarity to all docs
similarities = [cosine_sim(query_embedding, doc) for doc in doc_embeddings]

# 4. Rank: Return top-k most similar
top_results = sorted(similarities, reverse=True)[:k]
```

## When to Use a Vector Database

- **Without DB (numpy/lists)**: <10k documents, simple use cases
- **With DB (ChromaDB)**: >10k documents, need persistence, filtering, or updates

## Next Steps
→ Lesson 06: Vector Search with ChromaDB (optional for larger datasets)
