"""
04 - Cross-Encoder Reranking
=============================
Improve retrieval accuracy by reranking initial results with a cross-encoder.

Key concept: Cross-encoders score query-document pairs more accurately than embeddings alone.

Book reference: hands_on_LLM.II.8
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

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    MISSING_DEPENDENCIES.append('sentence_transformers')

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

# Cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def index_jobs(jobs: list[dict]):
    """Index jobs into vector database."""
    documents = [f"{j['title']} at {j['companyName']}\n{j['description'][:500]}" for j in jobs]
    ids = [j['id'] for j in jobs]
    metadatas = [{"title": j['title'], "company": j['companyName']} for j in jobs]
    collection.add(documents=documents, ids=ids, metadatas=metadatas)


def retrieve_and_rerank(query: str, initial_k: int = 20, final_k: int = 5) -> list[dict]:
    """Retrieve many candidates, then rerank with cross-encoder."""
    # 1. Initial retrieval with embeddings (cast wide net)
    results = collection.query(query_texts=[query], n_results=initial_k)

    # 2. Prepare query-document pairs for reranking
    pairs = [(query, doc) for doc in results["documents"][0]]

    # 3. Score pairs with cross-encoder
    scores = reranker.predict(pairs)

    # 4. Sort by reranker scores
    scored_docs = [
        {
            "text": doc,
            "metadata": meta,
            "embedding_score": dist,
            "rerank_score": float(score)
        }
        for doc, meta, dist, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            scores
        )
    ]

    # Sort by rerank score (descending)
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_docs[:final_k]


if __name__ == "__main__":
    # Index sample jobs
    jobs = load_sample_jobs(100)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")

    # Test query
    query = "Python developer with machine learning experience"

    print("=== Retrieval with Reranking ===")
    results = retrieve_and_rerank(query, initial_k=20, final_k=5)

    print(f"Query: {query}\n")
    print("Top 5 after reranking:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc['metadata']['title']} at {doc['metadata']['company']}")
        print(f"    Embedding score: {doc['embedding_score']:.4f}")
        print(f"    Rerank score: {doc['rerank_score']:.4f}")
        print(f"    {doc['text'][:150]}...")
