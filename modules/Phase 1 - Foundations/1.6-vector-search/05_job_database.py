"""
05 - Job Database
=================
Build a searchable job database with ChromaDB.

Key concept: Index your full job dataset for semantic search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils.data_loader import load_sample_jobs

# Setup
DB_PATH = Path(__file__).parent / "job_db"
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(
    name="job_posts",
    embedding_function=openai_ef,
)

# Load and index jobs (only if empty)
if collection.count() == 0:
    print("Indexing jobs...")
    jobs = load_sample_jobs(100)  # Start with 100 for demo
    
    # Prepare data
    documents = []
    ids = []
    metadatas = []
    
    for job in jobs:
        # Create searchable text
        doc_text = f"{job['title']} at {job['companyName']}\n{job['description'][:1000]}"
        documents.append(doc_text)
        ids.append(job['id'])
        metadatas.append({
            "title": job['title'],
            "company": job['companyName'],
            "location": job.get('location', ''),
            "level": job.get('experienceLevel', ''),
            "sector": job.get('sector', ''),
        })
    
    # Add in batches (ChromaDB limit is ~5000 per batch)
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Indexed {collection.count()} jobs")
else:
    print(f"Database already has {collection.count()} jobs")


def search_jobs(query: str, n_results: int = 5, **filters):
    """Search jobs with optional filters."""
    where = None
    if filters:
        conditions = [
            {k: v} for k, v in filters.items() if v
        ]
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )
    
    return [
        {
            "title": meta["title"],
            "company": meta["company"],
            "location": meta["location"],
            "score": 1 - dist,  # Convert distance to similarity
        }
        for meta, dist in zip(results["metadatas"][0], results["distances"][0])
    ]


# Demo searches
print("\n" + "="*60)
print("🔍 Search: Python data engineering")
for job in search_jobs("Python data engineering"):
    print(f"  {job['score']:.3f} | {job['title']} at {job['company']}")

print("\n" + "="*60)
print("🔍 Search: Business analyst (Amsterdam only)")
for job in search_jobs("Business analyst", location="Amsterdam, North Holland, Netherlands"):
    print(f"  {job['score']:.3f} | {job['title']} at {job['company']}")

print(f"\nDatabase stored at: {DB_PATH}")
