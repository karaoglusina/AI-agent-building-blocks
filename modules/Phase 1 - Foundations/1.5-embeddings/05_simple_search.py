"""
05 - Simple Vector Search
=========================
Search through documents using embeddings (no database needed).

Key concept: Compare query embedding to all document embeddings.
"""

import numpy as np
from openai import OpenAI

client = OpenAI()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity."""
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, documents: list[str], doc_embeddings: list, top_k: int = 3):
    """Search documents by semantic similarity."""
    # Embed the query
    query_embedding = get_embeddings([query])[0]
    
    # Calculate similarity to each document
    similarities = [
        (i, cosine_similarity(query_embedding, doc_emb))
        for i, doc_emb in enumerate(doc_embeddings)
    ]
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return [(documents[i], score) for i, score in similarities[:top_k]]


# Sample documents (job titles)
documents = [
    "Senior Python Developer - Backend",
    "Data Scientist with Machine Learning",
    "Frontend Engineer - React/TypeScript",
    "DevOps Engineer - Kubernetes/AWS",
    "Java Backend Developer - Microservices",
    "Full Stack Developer - Node.js",
    "Data Engineer - ETL Pipelines",
    "ML Engineer - Deep Learning",
    "Product Manager - Technical",
    "UX Designer - Mobile Apps",
]

# Pre-compute document embeddings (do this once!)
print("Embedding documents...")
doc_embeddings = get_embeddings(documents)
print(f"Embedded {len(documents)} documents\n")

# Search!
queries = [
    "I want to work with Python and databases",
    "Looking for frontend work",
    "AI and machine learning roles",
]

for query in queries:
    print(f"Query: {query}")
    print("-" * 40)
    results = search(query, documents, doc_embeddings, top_k=3)
    for doc, score in results:
        print(f"  {score:.3f} | {doc}")
    print()
