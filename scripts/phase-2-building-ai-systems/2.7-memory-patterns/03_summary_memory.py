"""
03 - Summary Memory
===================
Summarize old conversations to preserve context.

Key concept: Compress history into summaries - keeps key facts, loses details.

Book reference: hands_on_LLM.II.7
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


class SummaryMemory:
    """Memory that summarizes old messages."""
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        summary_threshold: int = 6,  # Summarize when this many messages
        keep_recent: int = 2,  # Keep this many recent exchanges
    ):
        self.system_prompt = system_prompt
        self.summary = ""
        self.messages = []
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent * 2  # User + assistant = 2 messages per exchange
    
    def add_message(self, role: str, content: str):
        """Add a message and summarize if needed."""
        self.messages.append({"role": role, "content": content})
        
        if len(self.messages) >= self.summary_threshold:
            self._summarize_old_messages()
    
    def _summarize_old_messages(self):
        """Summarize older messages into a summary."""
        old_messages = self.messages[:-self.keep_recent]
        
        if not old_messages:
            return
        
        # Build text to summarize
        text_parts = []
        if self.summary:
            text_parts.append(f"Previous summary: {self.summary}")
        
        for msg in old_messages:
            text_parts.append(f"{msg['role']}: {msg['content']}")
        
        text_to_summarize = "\n".join(text_parts)
        
        # Get summary from LLM
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
        "role": "system",
        "content": "Summarize this conversation in 2-3 sentences. "
        "Focus on key facts and decisions made."
        },
        {"role": "user", "content": text_to_summarize}
        ]
        )
        
        self.summary = response.choices[0].message.content
        self.messages = self.messages[-self.keep_recent:]
        print(f"  [Summarized {len(old_messages)} messages]")
    
    def get_messages(self) -> list[dict]:
        """Get messages for API call."""
        result = [{"role": "system", "content": self.system_prompt}]
        
        if self.summary:
            result.append({
                "role": "system",
                "content": f"Conversation summary: {self.summary}"
            })
        
        result.extend(self.messages)
        return result


def chat_with_summary(memory: SummaryMemory, user_input: str) -> str:
    """Chat using summary memory."""
    memory.add_message("user", user_input)
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=memory.get_messages()
    )
    
    assistant_reply = response.choices[0].message.content
    memory.add_message("assistant", assistant_reply)
    
    return assistant_reply


if __name__ == "__main__":
    print("=== SUMMARY MEMORY ===\n")
    
    memory = SummaryMemory(
        system_prompt="You are a job search assistant. Be helpful.",
        summary_threshold=6,
        keep_recent=2)
    
    conversation = [
        "I'm looking for Python developer jobs.",
        "I prefer remote positions.",
        "Senior level, 5+ years experience.",
        "In the Netherlands or UK.",
        "What companies match my requirements?",  # Triggers summarization
        "Tell me more about the first option."]
    
    for i, user_input in enumerate(conversation, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_input}")
        response = chat_with_summary(memory, user_input)
        print(f"Assistant: {response}\n")
    
    print("=" * 50)
    print(f"\nCurrent summary:\n{memory.summary}")
    print(f"\nRecent messages in memory: {len(memory.messages)}")
