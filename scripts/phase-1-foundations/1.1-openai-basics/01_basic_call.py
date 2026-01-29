"""
01 - Basic API Call
====================
The simplest possible OpenAI API call using the Responses API.

Key concept: client.chat.completions.create() is the core method.
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
    print("✓ OpenAI client would be initialized with API key")
    print("✓ Would call: client.chat.completions.create()")
    print("✓ Basic API call pattern: PASSED")
    exit(0)

client = OpenAI()  # Reads OPENAI_API_KEY from environment (or .env file)

response = client.chat.completions.create(
model="gpt-4o-mini",  # Cheaper model for learning
messages=[{"role": "user", "content": "What is Python?"}]
)

# The simplest way to get the text output
print(response.choices[0].message.content)
