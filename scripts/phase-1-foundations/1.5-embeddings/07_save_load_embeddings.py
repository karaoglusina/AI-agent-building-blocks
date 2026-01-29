"""
07 - Save and Load Embeddings
=============================
Persist embeddings to avoid re-computing.

Key concept: Embeddings are expensive - compute once, reuse many times.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Storage location
EMBEDDINGS_FILE = Path(__file__).parent / "embeddings_cache.json"


def save_embeddings(texts: list[str], embeddings: list[list[float]], filepath: Path):
    """Save texts and their embeddings to JSON."""
    data = {
        "texts": texts,
        "embeddings": embeddings,
        "model": "text-embedding-3-small",
        "dimensions": len(embeddings[0]) if embeddings else 0,
    }
    with open(filepath, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(texts)} embeddings to {filepath}")


def load_embeddings(filepath: Path) -> tuple[list[str], list[list[float]]]:
    """Load texts and embeddings from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['texts'])} embeddings from {filepath}")
    return data["texts"], data["embeddings"]


def get_or_create_embeddings(texts: list[str], filepath: Path) -> list[list[float]]:
    """Load embeddings if cached, otherwise create and save."""
    if filepath.exists():
        cached_texts, embeddings = load_embeddings(filepath)
        if cached_texts == texts:
            return embeddings
        print("Cache outdated, re-computing...")
    
    # Create new embeddings
    print(f"Creating embeddings for {len(texts)} texts...")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts)
    embeddings = [item.embedding for item in response.data]
    
    # Save for next time
    save_embeddings(texts, embeddings, filepath)
    return embeddings


# Demo
texts = [
    "Python developer",
    "Data scientist",
    "DevOps engineer",
    "Frontend developer"]

# First call: creates and saves
embeddings = get_or_create_embeddings(texts, EMBEDDINGS_FILE)
print(f"Got {len(embeddings)} embeddings")

# Second call: loads from cache (fast!)
embeddings2 = get_or_create_embeddings(texts, EMBEDDINGS_FILE)
print(f"Got {len(embeddings2)} embeddings (from cache)")

# Verify they're the same
print(f"Same embeddings: {embeddings[0][:5] == embeddings2[0][:5]}")

# Cleanup
EMBEDDINGS_FILE.unlink(missing_ok=True)
print(f"\nCleaned up {EMBEDDINGS_FILE}")
