"""
02 - Conversation Class
=======================
Encapsulate conversation state in a reusable class.

Key concept: Clean abstraction for managing conversations.
"""

from openai import OpenAI


class Conversation:
    """Manages a conversation with OpenAI."""
    
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def chat(self, user_input: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.responses.create(
            model=self.model,
            input=self.messages,
        )
        
        assistant_message = response.output_text
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
