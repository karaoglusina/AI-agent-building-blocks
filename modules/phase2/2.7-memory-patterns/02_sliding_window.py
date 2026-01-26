"""
02 - Sliding Window Memory
==========================
Keep only recent N messages.

Key concept: Bounded memory - loses old context but stays within token limits.

Book reference: hands_on_LLM.II.7
"""

import tiktoken
from openai import OpenAI

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-4o-mini")


class SlidingWindowMemory:
    """Memory that keeps only the most recent messages."""
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_messages: int = 10,
        max_tokens: int = None
    ):
        self.system_message = {"role": "system", "content": system_prompt}
        self.messages = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        """Add a message and trim if necessary."""
        self.messages.append({"role": role, "content": content})
        self._trim()
    
    def _trim(self):
        """Trim messages to fit within limits."""
        # Trim by message count
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Trim by token count if specified
        if self.max_tokens:
            while self._count_tokens() > self.max_tokens and len(self.messages) > 2:
                self.messages.pop(0)
    
    def _count_tokens(self) -> int:
        """Count tokens in current messages."""
        total = len(encoding.encode(self.system_message["content"]))
        for msg in self.messages:
            total += len(encoding.encode(msg["content"])) + 4  # overhead
        return total
    
    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        return [self.system_message] + self.messages
    
    def __len__(self):
        return len(self.messages)


def chat_with_sliding_window(memory: SlidingWindowMemory, user_input: str) -> str:
    """Chat using sliding window memory."""
    memory.add_message("user", user_input)
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=memory.get_messages()
    )
    
    assistant_reply = response.output_text
    memory.add_message("assistant", assistant_reply)
    
    return assistant_reply


if __name__ == "__main__":
    print("=== SLIDING WINDOW MEMORY ===\n")
    
    # Create memory with small window for demo
    memory = SlidingWindowMemory(
        system_prompt="You are a job search assistant. Be concise.",
        max_messages=4,  # Keep only last 4 messages
    )
    
    # Simulate long conversation
    conversation = [
        "I'm looking for Python jobs.",           # Will be forgotten
        "I prefer remote work.",                  # Will be forgotten
        "Senior level positions please.",         # Will be forgotten
        "In Amsterdam if possible.",              # Kept
        "What companies are hiring?",             # Kept
        "Tell me about the first one.",           # Kept
        "What were my original requirements?",    # Tests that old messages are gone
    ]
    
    for i, user_input in enumerate(conversation, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_input}")
        response = chat_with_sliding_window(memory, user_input)
        print(f"Assistant: {response}")
        print(f"Messages in memory: {len(memory)}\n")
    
    print("=" * 50)
    print("Final memory contents:")
    for msg in memory.get_messages()[1:]:  # Skip system
        role = msg["role"]
        content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"  [{role}] {content}")
    
    print("\nNote: Early messages about Python/remote/senior were dropped!")
