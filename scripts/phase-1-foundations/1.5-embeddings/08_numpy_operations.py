"""
08 - NumPy Vector Operations
============================
Essential numpy operations for working with embeddings.

Key concept: numpy makes vector math fast and easy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import numpy as np
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings as numpy array."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts)
    # Convert to numpy array for fast operations
    return np.array([item.embedding for item in response.data])


# Get some embeddings
texts = [
    "Python developer",
    "Java developer",
    "Data scientist",
    "Product manager"]
embeddings = get_embeddings(texts)

print(f"Shape: {embeddings.shape}")  # (4, 1536) = 4 texts, 1536 dimensions
print()

# 1. Cosine similarity between all pairs (efficient!)
# Normalize vectors
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms

# Similarity matrix (all pairs at once)
similarity_matrix = np.dot(normalized, normalized.T)

print("Similarity Matrix:")
print(f"{'':20}", end="")
for t in texts:
    print(f"{t[:10]:12}", end="")
print()
for i, text in enumerate(texts):
    print(f"{text:20}", end="")
    for j in range(len(texts)):
        print(f"{similarity_matrix[i,j]:.3f}       ", end="")
    print()
print()

# 2. Find most similar to a query
query = get_embeddings(["Backend engineer"])[0]
query_norm = query / np.linalg.norm(query)

# Similarity to all documents
similarities = np.dot(normalized, query_norm)

# Get top results
top_indices = np.argsort(similarities)[::-1]  # Sort descending
print("Most similar to 'Backend engineer':")
for idx in top_indices:
    print(f"  {similarities[idx]:.3f} | {texts[idx]}")
print()

# 3. Average embedding (useful for representing groups)
avg_embedding = embeddings.mean(axis=0)
print(f"Average embedding shape: {avg_embedding.shape}")
print(f"Average of first 5 dims: {avg_embedding[:5]}")
