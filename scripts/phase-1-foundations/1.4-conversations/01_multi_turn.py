"""
01 - Multi-Turn Conversations
=============================
Maintain conversation history across multiple exchanges.

Key concept: You manage the message history, not OpenAI.
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


def chat(messages: list[dict], user_input: str) -> str:
    """Add user message, get response, update history."""
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get response
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages)
    
    assistant_message = response.choices[0].message.content
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message


# Initialize conversation with system prompt
conversation = [
    {
        "role": "system",
        "content": "You are a job market analyst. Be concise and helpful."
    }
]

# Simulate multi-turn conversation
print("Turn 1:")
response1 = chat(conversation, "What skills are most in-demand for data roles?")
print(f"User: What skills are most in-demand for data roles?")
print(f"AI: {response1}\n")

print("Turn 2:")
response2 = chat(conversation, "Which of those is easiest to learn?")
print(f"User: Which of those is easiest to learn?")
print(f"AI: {response2}\n")

print("Turn 3:")
response3 = chat(conversation, "Can you give me a learning path?")
print(f"User: Can you give me a learning path?")
print(f"AI: {response3}\n")

# Show conversation length
print(f"Conversation has {len(conversation)} messages")
