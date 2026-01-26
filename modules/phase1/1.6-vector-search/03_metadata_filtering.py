"""
03 - Metadata Filtering
=======================
Filter search results by metadata.

Key concept: Combine semantic search with structured filters.
"""

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="jobs")

# Add documents with rich metadata
collection.add(
    documents=[
        "Senior Python Developer for fintech startup",
        "Junior Data Analyst for marketing team",
        "Mid-level Java Developer for enterprise",
        "Senior ML Engineer for AI research",
        "Junior Frontend Developer with React",
        "Senior DevOps Engineer for cloud platform",
    ],
    ids=[f"job{i}" for i in range(6)],
    metadatas=[
        {"level": "senior", "domain": "fintech", "location": "Amsterdam"},
        {"level": "junior", "domain": "marketing", "location": "Rotterdam"},
        {"level": "mid", "domain": "enterprise", "location": "Amsterdam"},
        {"level": "senior", "domain": "ai", "location": "Utrecht"},
        {"level": "junior", "domain": "web", "location": "Amsterdam"},
        {"level": "senior", "domain": "cloud", "location": "Rotterdam"},
    ],
)

# Query 1: No filter
print("All results for 'developer':")
results = collection.query(query_texts=["developer"], n_results=3)
for doc in results["documents"][0]:
    print(f"  {doc}")

# Query 2: Filter by level
print("\nSenior positions only:")
results = collection.query(
    query_texts=["developer"],
    n_results=3,
    where={"level": "senior"},  # Metadata filter
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc} [{meta['level']}]")

# Query 3: Filter by location
print("\nAmsterdam jobs:")
results = collection.query(
    query_texts=["developer"],
    n_results=3,
    where={"location": "Amsterdam"},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc} [{meta['location']}]")

# Query 4: Complex filter (AND)
print("\nSenior + Amsterdam:")
results = collection.query(
    query_texts=["developer"],
    n_results=3,
    where={"$and": [{"level": "senior"}, {"location": "Amsterdam"}]},
)
for doc in results["documents"][0]:
    print(f"  {doc}")

# Query 5: OR filter
print("\nAmsterdam OR Rotterdam:")
results = collection.query(
    query_texts=["engineer"],
    n_results=3,
    where={"$or": [{"location": "Amsterdam"}, {"location": "Rotterdam"}]},
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc} [{meta['location']}]")
