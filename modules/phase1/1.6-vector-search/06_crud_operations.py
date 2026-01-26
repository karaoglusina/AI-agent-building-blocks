"""
06 - CRUD Operations
====================
Create, Read, Update, Delete operations in ChromaDB.

Key concept: Manage your vector database like a regular database.
"""

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="crud_demo")

# CREATE - Add documents
print("=== CREATE ===")
collection.add(
    documents=["Python developer", "Java developer", "Data scientist"],
    ids=["1", "2", "3"],
    metadatas=[
        {"skills": "python,django", "level": "senior"},
        {"skills": "java,spring", "level": "mid"},
        {"skills": "python,ml", "level": "senior"},
    ],
)
print(f"Added 3 documents. Count: {collection.count()}")

# READ - Get by ID
print("\n=== READ ===")
result = collection.get(ids=["1", "2"])
print(f"Get by IDs: {result['documents']}")

# Read all
all_docs = collection.get()
print(f"All documents: {all_docs['documents']}")

# UPDATE - Modify existing
print("\n=== UPDATE ===")
collection.update(
    ids=["1"],
    documents=["Senior Python Developer with 10 years experience"],
    metadatas=[{"skills": "python,django,fastapi", "level": "senior"}],
)
updated = collection.get(ids=["1"])
print(f"Updated: {updated['documents'][0]}")

# UPSERT - Update or insert
print("\n=== UPSERT ===")
collection.upsert(
    ids=["3", "4"],  # 3 exists, 4 is new
    documents=["Data scientist (updated)", "New DevOps engineer"],
    metadatas=[
        {"skills": "python,ml,dl", "level": "senior"},
        {"skills": "aws,k8s", "level": "mid"},
    ],
)
print(f"After upsert. Count: {collection.count()}")

# DELETE - Remove documents
print("\n=== DELETE ===")
collection.delete(ids=["2"])
print(f"Deleted id=2. Count: {collection.count()}")

# Delete by filter
collection.delete(where={"level": "mid"})
print(f"Deleted mid-level. Count: {collection.count()}")

# Final state
print("\n=== FINAL STATE ===")
final = collection.get()
for doc, meta in zip(final['documents'], final['metadatas']):
    print(f"  {doc[:50]}... | {meta}")
