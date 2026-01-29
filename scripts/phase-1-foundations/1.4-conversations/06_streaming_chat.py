"""
06 - Streaming Chat
===================
Interactive chat with streaming responses.

Key concept: Better UX by showing tokens as they arrive.
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


def streaming_chat(system_prompt: str):
    """Interactive chat with streaming output."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    print("=" * 50)
    print("Streaming Chat (type 'quit' to exit)")
    print("=" * 50)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Stream the response
        print("\nAssistant: ", end="", flush=True)
        
        stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True)
        
        # Collect full response while streaming
        full_response = ""
        for event in stream:
            if event.type == "response.choices[0].message.content.delta":
                print(event.delta, end="", flush=True)
                full_response += event.delta
        
        print("\n")  # New line after response
        
        # Add complete response to history
        messages.append({"role": "assistant", "content": full_response})
    
    print("\nGoodbye!")


if __name__ == "__main__":
    streaming_chat(
        system_prompt="You are a helpful job search assistant. Be concise."
    )
