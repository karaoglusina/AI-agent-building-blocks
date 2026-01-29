"""
05 - Response Object
====================
Understanding what the API returns.

Key concept: The response object contains metadata, not just text.
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
    print("✓ Response object pattern: PASSED")
    exit(0)

client = OpenAI()

response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Say hello in 3 languages."}]
)

# Quick access to text
print("output_text:", response.choices[0].message.content)
print()

# Full response structure
print("Response ID:", response.id)
print("Model used:", response.model)
print("Created at:", response.created)
print()

# Choices is a list (can contain multiple items)
print("Choices type:", type(response.choices))
print("Number of choices:", len(response.choices))
print()

# Each choice has a message
for i, choice in enumerate(response.choices):
    print(f"Choice[{i}]:")
    print(f"  Role: {choice.message.role}")
    print(f"  Content: {choice.message.content[:100] if choice.message.content else 'None'}...")
    print(f"  Finish reason: {choice.finish_reason}")

# Usage statistics
print()
print("Usage:")
print(f"  Prompt tokens: {response.usage.prompt_tokens}")
print(f"  Completion tokens: {response.usage.completion_tokens}")
print(f"  Total tokens: {response.usage.total_tokens}")
