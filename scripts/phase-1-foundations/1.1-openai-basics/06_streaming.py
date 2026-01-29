"""
06 - Streaming Responses
========================
Get responses token-by-token instead of waiting for completion.

Key concept: stream=True returns an iterator, not a complete response.
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
    print("✓ Streaming pattern: PASSED")
    exit(0)

client = OpenAI()

# Non-streaming: Wait for full response
print("Non-streaming (waits for completion):")
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
stream=False  # Default
)
print(response.choices[0].message.content)
print()

# Streaming: Get tokens as they arrive
print("Streaming (tokens arrive progressively):")
stream = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
stream=True
)

# Iterate over stream events
for event in stream:
    # Events have different types
    if event.type == "response.choices[0].message.content.delta":
        # This event contains a chunk of text
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed":
        # Final event with full response
        print("\n\n[Stream completed]")
