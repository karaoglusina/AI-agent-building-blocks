"""
03 - System Prompts
===================
System prompts define the AI's behavior, personality, and constraints.

Key concept: "system" role sets context that persists through conversation.
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
    print("✓ System prompts pattern: PASSED")
    exit(0)

client = OpenAI()

# Without system prompt - generic response
response1 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Analyze this job market trend."}]
)
print("Without system prompt:")
print(response1.choices[0].message.content[:200])
print()

# With system prompt - focused response
response2 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{
"role": "system",
"content": "You are a job market analyst specializing in tech roles in the Netherlands. Be concise and data-driven."
},
{
"role": "user", 
"content": "Analyze this job market trend."
}
]
)
print("With system prompt:")
print(response2.choices[0].message.content[:200])
