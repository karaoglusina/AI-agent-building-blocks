"""
04 - Handling No Results
========================
Graceful fallback when retrieval fails.

Key concept: Always have a fallback - empty results shouldn't crash or confuse.

Book reference: AI_eng.6
"""

from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="limited_jobs", embedding_function=openai_ef)

# Index only Python jobs (limited corpus)
jobs = load_sample_jobs(20)
for job in jobs:
    if "python" in job["title"].lower():
        collection.add(
            documents=[job["description"][:500]],
            ids=[job["id"]],
            metadatas=[{"title": job["title"]}]
        )

RELEVANCE_THRESHOLD = 0.5  # Minimum similarity score


def retrieve_with_threshold(query: str, threshold: float = RELEVANCE_THRESHOLD) -> list[dict]:
    """Retrieve documents, filtering by relevance threshold."""
    results = collection.query(query_texts=[query], n_results=5)
    
    if not results["documents"][0]:
        return []
    
    # Filter by threshold (convert distance to similarity)
    filtered = []
    for doc, meta, dist in zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    ):
        similarity = 1 - dist  # Convert distance to similarity
        if similarity >= threshold:
            filtered.append({"text": doc, "title": meta["title"], "score": similarity})
    
    return filtered


def generate_response(query: str, docs: list[dict]) -> str:
    """Generate response with fallback handling."""
    
    if not docs:
        # No relevant results - provide helpful fallback
        return client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": "The user asked about jobs but no matching results were found. "
                               "Acknowledge this politely and suggest they broaden their search."
                },
                {"role": "user", "content": query}
            ]
        ).output_text
    
    # Normal RAG response
    context = "\n\n".join([f"[{d['score']:.0%}] {d['text']}" for d in docs])
    return client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Answer based on the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    ).output_text


if __name__ == "__main__":
    print(f"=== NO RESULTS HANDLING (Indexed: {collection.count()} Python jobs) ===\n")
    
    queries = [
        "Python developer positions",     # Should find results
        "Quantum computing researcher",   # Unlikely to find
        "Java Spring Boot developer",     # Not in corpus
    ]
    
    for query in queries:
        docs = retrieve_with_threshold(query)
        print(f"Q: {query}")
        print(f"   Retrieved: {len(docs)} relevant docs")
        
        response = generate_response(query, docs)
        print(f"A: {response}\n")
        print("-" * 60 + "\n")
