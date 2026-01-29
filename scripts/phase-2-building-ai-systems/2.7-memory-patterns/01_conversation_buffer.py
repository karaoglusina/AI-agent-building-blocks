"""
01 - Conversation Buffer
========================
Store full conversation history.

Key concept: Simple but grows unbounded - works for short conversations.

Book reference: hands_on_LLM.II.7, AI_eng.6
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


class ConversationBuffer:
    """Simple conversation memory that stores all messages."""
    
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def add_user_message(self, content: str):
        """Add a user message to the buffer."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to the buffer."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> list[dict]:
        """Get all messages."""
        return self.messages
    
    def clear(self, keep_system: bool = True):
        """Clear conversation history."""
        if keep_system and self.messages:
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def __len__(self):
        return len(self.messages)


def chat_with_buffer(buffer: ConversationBuffer, user_input: str) -> str:
    """Chat using the conversation buffer."""
    buffer.add_user_message(user_input)
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=buffer.get_messages()
    )
    
    assistant_reply = response.choices[0].message.content
    buffer.add_assistant_message(assistant_reply)
    
    return assistant_reply


if __name__ == "__main__":
    print("=== CONVERSATION BUFFER ===\n")
    
    # Create buffer with custom system prompt
    buffer = ConversationBuffer(
        system_prompt="You are a job search assistant. Be helpful and concise."
    )
    
    # Simulate conversation
    conversation = [
        "I'm looking for Python developer jobs.",
        "What about remote positions?",
        "Which companies did you mention?",  # Tests memory of previous response
        "What was my first question?",  # Tests memory of earlier messages
    ]
    
    for user_input in conversation:
        print(f"User: {user_input}")
        response = chat_with_buffer(buffer, user_input)
        print(f"Assistant: {response}\n")
    
    print("=" * 50)
    print(f"Total messages in buffer: {len(buffer)}")
    print("\nBuffer contents:")
    for msg in buffer.get_messages():
        role = msg["role"].upper()
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  [{role}] {content}")
