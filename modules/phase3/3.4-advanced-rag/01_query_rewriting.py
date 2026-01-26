"""
01 - Query Rewriting
====================
Transform user queries for better retrieval results.

Key concept: Query rewriting improves retrieval by expanding, clarifying, or decomposing queries.

Book reference: AI_eng.6, hands_on_LLM.II.8
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


def rewrite_query(query: str) -> list[str]:
    """Rewrite query into multiple variations for better retrieval."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are a query expansion expert. Given a job search query, "
                           "rewrite it into 3 variations that capture different aspects. "
                           "Return ONLY the 3 queries, one per line, no numbering or explanations."
            },
            {
                "role": "user",
                "content": f"Original query: {query}"
            }
        ]
    )

    # Parse multiple queries from response
    queries = [q.strip() for q in response.output_text.strip().split('\n') if q.strip()]
    return queries[:3]  # Ensure max 3 queries


def retrieve_with_rewriting(query: str, n_results: int = 3) -> list[dict]:
    """Retrieve using query rewriting."""
    # 1. Rewrite query into variations
    rewritten_queries = rewrite_query(query)
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten_queries}\n")

    # 2. Retrieve with each variation
    all_docs = {}
    for q in rewritten_queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for doc, meta, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0]
        ):
            if doc_id not in all_docs:
                all_docs[doc_id] = {"text": doc, "metadata": meta, "count": 0}
            all_docs[doc_id]["count"] += 1

    # 3. Sort by frequency (how many queries retrieved it)
    sorted_docs = sorted(all_docs.values(), key=lambda x: x["count"], reverse=True)
    return sorted_docs[:n_results]


if __name__ == "__main__":
    # Index sample jobs
    jobs = load_sample_jobs(50)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")

    # Test queries
    test_queries = [
        "ML engineer job",
        "remote work",
        "good salary startup",
    ]

    for query in test_queries:
        print("=" * 60)
        docs = retrieve_with_rewriting(query)
        print(f"Retrieved {len(docs)} unique documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n[{i}] (retrieved {doc['count']}x)")
            print(f"{doc['metadata']['title']} at {doc['metadata']['company']}")
            print(f"{doc['text'][:150]}...")
        print("\n")
