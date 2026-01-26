"""
06 - RAG Fusion
===============
Combine results from multiple query variations using reciprocal rank fusion.

Key concept: Multiple perspectives on the same query improve retrieval robustness.

Book reference: AI_eng.6
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Setup
client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="jobs", embedding_function=openai_ef)


def index_jobs(jobs: list[dict]):
    """Index jobs into vector database."""
    documents = [f"{j['title']} at {j['companyName']}\n{j['description'][:500]}" for j in jobs]
    ids = [j['id'] for j in jobs]
    metadatas = [{"title": j['title'], "company": j['companyName']} for j in jobs]
    collection.add(documents=documents, ids=ids, metadatas=metadatas)


def generate_query_variations(query: str, n: int = 3) -> list[str]:
    """Generate multiple variations of the query."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Generate {n} different search queries that capture different aspects "
                           f"of the user's intent. Return ONLY the queries, one per line."
            },
            {"role": "user", "content": query}
        ]
    )

    queries = [q.strip("- ").strip() for q in response.choices[0].message.content.split("\n") if q.strip()]
    return [query] + queries[:n-1]  # Include original + variations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    Combine multiple rankings using reciprocal rank fusion.

    RRF(d) = sum over all rankings r: 1 / (k + rank_r(d))
    """
    fusion_scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = 0
            fusion_scores[doc_id] += 1 / (k + rank)

    return fusion_scores


def rag_fusion(query: str, n_results: int = 5) -> list[dict]:
    """Retrieve using RAG Fusion: multiple queries + reciprocal rank fusion."""
    # 1. Generate query variations
    queries = generate_query_variations(query, n=3)
    print(f"Original: {query}")
    print(f"Variations: {queries[1:]}\n")

    # 2. Retrieve with each query
    rankings = []
    all_docs = {}

    for q in queries:
        results = collection.query(query_texts=[q], n_results=n_results * 2)
        rankings.append(results["ids"][0])

        # Store document details
        for doc_id, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        ):
            if doc_id not in all_docs:
                all_docs[doc_id] = {"text": doc, "metadata": meta}

    # 3. Fuse rankings
    fusion_scores = reciprocal_rank_fusion(rankings)

    # 4. Sort by fusion score
    sorted_ids = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)

    # 5. Return top results
    results = []
    for doc_id, score in sorted_ids[:n_results]:
        doc = all_docs[doc_id]
        doc["fusion_score"] = score
        doc["id"] = doc_id
        results.append(doc)

    return results


if __name__ == "__main__":
    # Index jobs
    jobs = load_sample_jobs(100)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")

    # Test query
    query = "machine learning engineer position"

    print("=== RAG Fusion ===")
    results = rag_fusion(query, n_results=5)

    print(f"Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc['metadata']['title']} at {doc['metadata']['company']}")
        print(f"    Fusion score: {doc['fusion_score']:.4f}")
        print(f"    {doc['text'][:120]}...")
