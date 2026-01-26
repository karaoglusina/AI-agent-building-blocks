"""
06 - Job Search with Embeddings
===============================
Search through real job data using embeddings.

Key concept: Apply vector search to your actual dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from openai import OpenAI
from utils.data_loader import load_sample_jobs

client = OpenAI()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch embed texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a, b) -> float:
    """Cosine similarity between vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Load jobs (small sample for demo)
jobs = load_sample_jobs(50)
print(f"Loaded {len(jobs)} jobs")

# Create searchable text for each job
# Combine key fields into one searchable string
searchable_texts = [
    f"{job['title']} at {job['companyName']} - {job['description'][:500]}"
    for job in jobs
]

# Embed all jobs
print("Embedding jobs...")
job_embeddings = get_embeddings(searchable_texts)
print("Done!\n")


def search_jobs(query: str, top_k: int = 5):
    """Search jobs by semantic similarity."""
    query_embedding = get_embeddings([query])[0]
    
    # Score all jobs
    scores = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(job_embeddings)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top results with job data
    results = []
    for idx, score in scores[:top_k]:
        results.append({
            "score": score,
            "title": jobs[idx]["title"],
            "company": jobs[idx]["companyName"],
            "location": jobs[idx]["location"],
            "snippet": jobs[idx]["description"][:150] + "..."
        })
    return results


# Demo searches
queries = [
    "Python data engineering role",
    "Business analyst with SQL",
    "Remote work cloud infrastructure",
]

for query in queries:
    print(f"🔍 Query: {query}")
    print("=" * 60)
    results = search_jobs(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['title']}")
        print(f"           at {r['company']} ({r['location']})")
    print()
