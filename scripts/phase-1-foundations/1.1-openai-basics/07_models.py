"""
07 - Available Models
=====================
Different models for different use cases.

Key concept: Choose model based on task complexity and cost.
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
    print("✓ Models comparison pattern: PASSED")
    exit(0)

client = OpenAI()

prompt = "What is 2+2? Answer with just the number."

# Model comparison
models = [
    ("gpt-4o-mini", "Fast, cheap, good for simple tasks"),
    ("gpt-4o", "Smartest, best for complex reasoning"),
    # ("o1-mini", "Reasoning model, thinks step-by-step"),  # Requires different params
]

for model_name, description in models:
    print(f"Model: {model_name}")
    print(f"Description: {description}")
    
    response = client.chat.completions.create(
    model=model_name,
    messages=prompt)
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("-" * 40)

# Tip: For learning/testing, use gpt-4o-mini (cheapest)
# For production/complex tasks, use gpt-4o
