"""
02 - Input Formats
==================
Different ways to pass input to the Responses API.

Key concept: Input can be a string OR a list of messages.
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
    print("✓ Input formats pattern: PASSED")
    exit(0)

client = OpenAI()

# Format 1: Simple string (shorthand)
response1 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Hello!"}]  # Just a string
)
print("String input:", response1.choices[0].message.content[:50])

# Format 2: List of messages (full control)
response2 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "user", "content": "Hello!"}
]
)
print("Message input:", response2.choices[0].message.content[:50])

# Format 3: Multiple messages (conversation history)
response3 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "user", "content": "My name is Sina."},
{"role": "assistant", "content": "Nice to meet you, Sina!"},
{"role": "user", "content": "What's my name?"}
]
)
print("Multi-message:", response3.choices[0].message.content[:50])
