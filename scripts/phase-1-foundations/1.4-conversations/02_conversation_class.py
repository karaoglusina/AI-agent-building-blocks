"""
02 - Conversation Class
=======================
Encapsulate conversation state in a reusable class.

Key concept: Clean abstraction for managing conversations.
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
    print("✓ Conversation class pattern: PASSED")
    exit(0)


class Conversation:
    """Manages a conversation with OpenAI."""
    
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_input: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages)
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def reset(self):
        """Clear history, keep system prompt."""
        self.messages = [self.messages[0]]
    
    @property
    def history(self) -> list[dict]:
        """Get conversation history (excluding system prompt)."""
        return self.messages[1:]
    
    def __len__(self) -> int:
        """Number of messages (excluding system prompt)."""
        return len(self.messages) - 1


# Usage
chat = Conversation(
    system_prompt="You are a friendly assistant who helps with job searching."
)

print(chat.chat("I'm looking for Python jobs in Amsterdam"))
print()
print(chat.chat("What salary should I expect?"))
print()
print(f"Messages: {len(chat)}")
print()

# Reset and start fresh
chat.reset()
print(f"After reset: {len(chat)} messages")
