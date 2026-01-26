"""
01 - ChromaDB Basics
====================
Introduction to ChromaDB - a local vector database.

Key concept: ChromaDB handles storage, indexing, and search for you.
"""

import chromadb

# Create a client (in-memory by default)
client = chromadb.Client()

# Create a collection (like a table)
collection = client.create_collection(name="demo")

# Add documents with IDs
collection.add(
    documents=[
        "Python developer with Django experience",
        "Java engineer specializing in microservices",
        "Data scientist with machine learning background",
        "Frontend developer with React expertise",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],  # Unique IDs required
)

print(f"Collection has {collection.count()} documents")

# Query the collection
results = collection.query(
    query_texts=["backend programming"],
    n_results=2,
)

print("\nQuery: 'backend programming'")
print("Results:")
for i, doc in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]
    doc_id = results["ids"][0][i]
    print(f"  {doc_id}: {doc} (distance: {distance:.4f})")

# Note: ChromaDB uses L2 (Euclidean) distance by default
# Lower distance = more similar
