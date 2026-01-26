"""
01 - Multi-Turn Conversations
=============================
Maintain conversation history across multiple exchanges.

Key concept: You manage the message history, not OpenAI.
"""

from openai import OpenAI

client = OpenAI()


def chat(messages: list[dict], user_input: str) -> str:
    """Add user message, get response, update history."""
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Get response
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
    )
    
    assistant_message = response.output_text
    
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
