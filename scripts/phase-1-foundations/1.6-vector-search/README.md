# Lesson 06: Vector Search with ChromaDB

## Overview
ChromaDB is a local vector database that handles storage, indexing, and semantic search. Use it when your dataset is too large for simple numpy arrays.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_chroma_basics.py` | Basic add and query | `python 01_chroma_basics.py` |
| `02_persistent_storage.py` | Save to disk | `python 02_persistent_storage.py` |
| `03_metadata_filtering.py` | Filter by metadata | `python 03_metadata_filtering.py` |
| `04_openai_embeddings.py` | Use OpenAI embeddings | `python 04_openai_embeddings.py` |
| `05_job_database.py` | Build job search database | `python 05_job_database.py` |
| `06_crud_operations.py` | Create, Read, Update, Delete | `python 06_crud_operations.py` |

## Installation

```bash
pip install chromadb
```

## Key Concepts

### Client Types
```python
# In-memory (for testing)
client = chromadb.Client()

# Persistent (saves to disk)
client = chromadb.PersistentClient(path="./chroma_data")
```

### Collections
```python
# Create or get
collection = client.get_or_create_collection(name="my_collection")

# With custom embeddings
collection = client.create_collection(
    name="my_collection",
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
)
```

### Adding Documents
```python
collection.add(
    documents=["text1", "text2"],
    ids=["id1", "id2"],
    metadatas=[{"key": "value"}, {"key": "value"}],
)
```

### Querying
```python
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    where={"field": "value"},  # Optional filter
)
```

## When to Use ChromaDB vs Numpy

| Approach | Use When |
|----------|----------|
| Numpy arrays | <10k docs, simple prototype, no persistence needed |
| ChromaDB | >10k docs, need persistence, need filtering, production use |

## Next Steps
→ Build your job market chatbot combining conversations + embeddings + tools!
