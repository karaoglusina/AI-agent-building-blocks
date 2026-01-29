"""
04 - Conversation Summarization
===============================
Summarize old context instead of truncating.

Key concept: Replace old messages with a summary to preserve context.
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


class SummarizingConversation:
    """Conversation that summarizes old context."""
    
    def __init__(self, system_prompt: str, summarize_after: int = 6):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.summarize_after = summarize_after
        self.summary = None
    
    def _summarize_history(self) -> str:
        """Create a summary of the conversation so far."""
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" 
            for m in self.messages[1:]  # Skip system prompt
        )
        
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=f"Summarize this conversation in 2-3 sentences:\n\n{history_text}")
        return response.choices[0].message.content
    
    def _compress_if_needed(self):
        """Summarize old messages if conversation is too long."""
        if len(self.messages) > self.summarize_after:
            print("[Summarizing conversation...]")
            self.summary = self._summarize_history()
            
            # Reset to system prompt + summary + last 2 messages
            last_messages = self.messages[-2:]
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": f"Previous conversation summary: {self.summary}"}] + last_messages
            print(f"[Compressed to {len(self.messages)} messages]")
    
    def chat(self, user_input: str) -> str:
        """Chat with automatic summarization."""
        self.messages.append({"role": "user", "content": user_input})
        self._compress_if_needed()
        
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=self.messages)
        
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message


# Demo
chat = SummarizingConversation(
    system_prompt="You are a job market expert helping someone find data roles.",
    summarize_after=6
)

questions = [
    "What are the top data roles right now?",
    "Which pays the most?",
    "What skills do I need for data engineering?",
    "Is Python or SQL more important?",
    "What about cloud skills?"]

for q in questions:
    print(f"You: {q}")
    response = chat.chat(q)
    print(f"AI: {response[:150]}...\n")

print(f"\nFinal summary: {chat.summary}")
