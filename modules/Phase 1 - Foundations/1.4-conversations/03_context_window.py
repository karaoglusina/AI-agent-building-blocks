"""
03 - Context Window Management
==============================
Handle conversations that exceed the model's context limit.

Key concept: Truncate or summarize old messages to fit context window.
"""

from openai import OpenAI

client = OpenAI()


def count_tokens_approx(text: str) -> int:
    """Rough token estimate (actual tokenization is more complex)."""
    # Rule of thumb: ~4 chars per token for English
    return len(text) // 4


class ManagedConversation:
    """Conversation with automatic context management."""
    
    def __init__(self, system_prompt: str, max_tokens: int = 4000):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_tokens = max_tokens
    
    def _get_total_tokens(self) -> int:
        """Estimate total tokens in conversation."""
        return sum(count_tokens_approx(m["content"]) for m in self.messages)
    
    def _truncate_if_needed(self):
        """Remove oldest messages if context is too large."""
        while self._get_total_tokens() > self.max_tokens and len(self.messages) > 2:
            # Keep system prompt (index 0), remove oldest user/assistant pair
            del self.messages[1:3]
            print(f"[Truncated: now {len(self.messages)} messages]")
    
    def chat(self, user_input: str) -> str:
        """Send message with automatic context management."""
        self.messages.append({"role": "user", "content": user_input})
        self._truncate_if_needed()
        
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=self.messages,
        )
        
        assistant_message = response.output_text
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message


# Test with artificially low limit
chat = ManagedConversation(
    system_prompt="You are a helpful assistant.",
    max_tokens=500  # Very low for demo
)

# Have a conversation that will exceed the limit
for i in range(5):
    response = chat.chat(f"Tell me an interesting fact #{i+1} about AI in job recruiting.")
    print(f"Response {i+1}: {response[:100]}...")
    print(f"Messages: {len(chat.messages)}, Tokens: {chat._get_total_tokens()}")
    print()
