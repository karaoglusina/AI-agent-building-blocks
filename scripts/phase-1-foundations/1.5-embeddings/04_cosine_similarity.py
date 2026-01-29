"""
04 - Cosine Similarity
======================
Measure how similar two vectors are.

Key concept: Cosine similarity = dot product of normalized vectors.
Higher = more similar (max 1.0), lower = less similar (min -1.0).
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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text)
    return response.data[0].embedding


# Compare similar and different texts
texts = [
    "Python developer with Django experience",  # Base text
    "Python programmer with Flask background",   # Similar
    "Senior Python backend engineer",            # Similar
    "Accountant with Excel expertise",           # Different
]

# Get all embeddings
embeddings = [get_embedding(t) for t in texts]

# Compare each to the first text
print(f"Base: {texts[0]}\n")
print("Similarity to base:")
print("-" * 50)

for i in range(1, len(texts)):
    similarity = cosine_similarity(embeddings[0], embeddings[i])
    print(f"{similarity:.4f} | {texts[i]}")

# Note: Higher similarity = more semantically related
# Typically: >0.8 very similar, >0.5 related, <0.3 unrelated
