"""
05 - Hybrid Search
==================
Combine BM25 keyword search with semantic vector search for better retrieval.

Key concept: Hybrid search catches both exact term matches and semantic similarity.

Book reference: AI_eng.6, hands_on_LLM.II.8
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import chromadb
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

try:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

from rank_bm25 import BM25Okapi
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Setup
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="jobs", embedding_function=openai_ef)


def index_jobs(jobs: list[dict]) -> tuple[list[str], BM25Okapi]:
    """Index jobs in both vector DB and BM25."""
    # Vector index
    documents = [f"{j['title']} at {j['companyName']}\n{j['description'][:500]}" for j in jobs]
    ids = [j['id'] for j in jobs]
    metadatas = [{"title": j['title'], "company": j['companyName']} for j in jobs]
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # BM25 index (tokenized)
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return documents, bm25


def hybrid_search(
    query: str,
    documents: list[str],
    bm25: BM25Okapi,
    k: int = 5,
    alpha: float = 0.5
) -> list[dict]:
    """
    Hybrid search combining BM25 and semantic search.

    Args:
        query: Search query
        documents: All documents
        bm25: BM25 index
        k: Number of results
        alpha: Weight for semantic search (1-alpha for BM25)
    """
    # 1. Semantic search scores
    semantic_results = collection.query(query_texts=[query], n_results=len(documents))
    semantic_scores = {}
    for doc_id, distance in zip(semantic_results["ids"][0], semantic_results["distances"][0]):
        # Convert distance to similarity (lower is better, so invert)
        semantic_scores[doc_id] = 1 / (1 + distance)

    # 2. BM25 scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to 0-1 range
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores = [score / max_bm25 for score in bm25_scores]

    # 3. Combine scores
    combined_results = []
    for idx, doc_id in enumerate(semantic_results["ids"][0]):
        semantic_score = semantic_scores.get(doc_id, 0)
        bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0

        combined_score = alpha * semantic_score + (1 - alpha) * bm25_score

        combined_results.append({
            "text": semantic_results["documents"][0][idx],
            "metadata": semantic_results["metadatas"][0][idx],
            "semantic_score": semantic_score,
            "bm25_score": bm25_score,
            "combined_score": combined_score
        })

    # Sort by combined score
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    return combined_results[:k]


if __name__ == "__main__":
    # Index jobs
    jobs = load_sample_jobs(100)
    documents, bm25 = index_jobs(jobs)
    print(f"Indexed {len(documents)} jobs\n")

    # Test query
    query = "senior Python developer"

    print("=== Hybrid Search (50% semantic, 50% BM25) ===")
    results = hybrid_search(query, documents, bm25, k=5, alpha=0.5)

    print(f"Query: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc['metadata']['title']} at {doc['metadata']['company']}")
        print(f"    Semantic: {doc['semantic_score']:.3f} | BM25: {doc['bm25_score']:.3f} | Combined: {doc['combined_score']:.3f}")
        print(f"    {doc['text'][:120]}...")
