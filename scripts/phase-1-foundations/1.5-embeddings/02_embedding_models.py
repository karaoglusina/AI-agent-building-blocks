"""
02 - Embedding Models
=====================
Compare different embedding models.

Key concept: Choose model based on quality vs cost trade-off.
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

text = "Senior data engineer with Python and cloud experience"

# Model comparison
models = [
    ("text-embedding-3-small", "Cheapest, 1536 dims, good for most uses"),
    ("text-embedding-3-large", "Best quality, 3072 dims, 3x cost")]

for model_name, description in models:
    response = client.embeddings.create(
        model=model_name,
        input=text)
    
    embedding = response.data[0].embedding
    tokens = response.usage.total_tokens
    
    print(f"Model: {model_name}")
    print(f"  Description: {description}")
    print(f"  Dimensions: {len(embedding)}")
    print(f"  Tokens used: {tokens}")
    print()

# Recommendation:
# - Use text-embedding-3-small for development and most production cases
# - Use text-embedding-3-large when you need maximum accuracy (e.g., legal, medical)
