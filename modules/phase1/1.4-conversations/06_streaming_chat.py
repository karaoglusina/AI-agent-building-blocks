"""
06 - Streaming Chat
===================
Interactive chat with streaming responses.

Key concept: Better UX by showing tokens as they arrive.
"""

from openai import OpenAI

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
        
        stream = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            stream=True,
        )
        
        # Collect full response while streaming
        full_response = ""
        for event in stream:
            if event.type == "response.output_text.delta":
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
