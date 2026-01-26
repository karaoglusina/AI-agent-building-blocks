"""
03 - Batch Embeddings
=====================
Embed multiple texts in a single API call.

Key concept: Batch for efficiency - fewer API calls, lower cost.
"""

from openai import OpenAI

client = OpenAI()

# Multiple texts to embed
texts = [
    "Python developer with Django experience",
    "Java engineer specializing in microservices",
    "Data scientist with machine learning background",
    "Frontend developer with React and TypeScript",
    "DevOps engineer with Kubernetes expertise",
]

# Single API call for all texts
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,  # Pass list of strings
)

# Results come in same order as input
print(f"Embedded {len(response.data)} texts")
print(f"Total tokens: {response.usage.total_tokens}")
print()

for i, item in enumerate(response.data):
    print(f"Text {i+1}: {texts[i][:40]}...")
    print(f"  Vector dims: {len(item.embedding)}")
    print(f"  First 5: {item.embedding[:5]}")
    print()

# Tip: OpenAI allows up to 2048 texts per batch
# For large datasets, batch in chunks of ~100-500 texts
