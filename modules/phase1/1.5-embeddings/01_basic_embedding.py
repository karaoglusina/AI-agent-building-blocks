"""
01 - Basic Embedding
====================
Create a vector representation of text.

Key concept: Embeddings map text to high-dimensional vectors.
"""

from openai import OpenAI

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
