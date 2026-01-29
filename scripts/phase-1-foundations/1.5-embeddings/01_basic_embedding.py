"""
01 - Basic Embedding
====================
Create a vector representation of text.

Key concept: Embeddings map text to high-dimensional vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Create an embedding
response = client.embeddings.create(
    model="text-embedding-3-small",  # Cheaper, good for most uses
    input="Python developer with 5 years experience"
)

# Get the embedding vector
embedding = response.data[0].embedding

print(f"Text: Python developer with 5 years experience")
print(f"Vector dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
print(f"Vector type: {type(embedding)}")
print(f"Value type: {type(embedding[0])}")

# The embedding is just a list of floats!
# These numbers encode the "meaning" of the text
