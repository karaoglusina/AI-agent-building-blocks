"""
02 - Persistent Storage
=======================
Save ChromaDB data to disk.

Key concept: Use PersistentClient to save between runs.
"""

import chromadb
from pathlib import Path

# Directory for storing data
DB_PATH = Path(__file__).parent / "chroma_data"

# Create persistent client
client = chromadb.PersistentClient(path=str(DB_PATH))

# Get or create collection
collection = client.get_or_create_collection(name="jobs")

# Check if already populated
if collection.count() == 0:
    print("Adding documents...")
    collection.add(
        documents=[
            "Senior Python Developer - Backend systems",
            "Data Engineer - ETL and pipelines",
            "ML Engineer - Model deployment",
            "DevOps Engineer - CI/CD and cloud",
        ],
        ids=["job1", "job2", "job3", "job4"],
        metadatas=[
            {"level": "senior", "tech": "python"},
            {"level": "mid", "tech": "sql"},
            {"level": "senior", "tech": "python"},
            {"level": "mid", "tech": "aws"},
        ],
    )
    print(f"Added {collection.count()} documents")
else:
    print(f"Collection already has {collection.count()} documents")

# Query
results = collection.query(
    query_texts=["Python programming"],
    n_results=2,
)

print("\nQuery results:")
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  {doc}")
    print(f"    metadata: {meta}, distance: {dist:.4f}")

print(f"\nData persisted to: {DB_PATH}")

# To clean up:
# import shutil
# shutil.rmtree(DB_PATH)
