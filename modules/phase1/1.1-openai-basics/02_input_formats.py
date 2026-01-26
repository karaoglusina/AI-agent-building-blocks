"""
02 - Input Formats
==================
Different ways to pass input to the Responses API.

Key concept: Input can be a string OR a list of messages.
"""

from openai import OpenAI

client = OpenAI()

# Format 1: Simple string (shorthand)
response1 = client.responses.create(
    model="gpt-4o-mini",
    input="Hello!"  # Just a string
)
print("String input:", response1.output_text[:50])

# Format 2: List of messages (full control)
response2 = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "user", "content": "Hello!"}
    ]
)
print("Message input:", response2.output_text[:50])

# Format 3: Multiple messages (conversation history)
response3 = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "user", "content": "My name is Sina."},
        {"role": "assistant", "content": "Nice to meet you, Sina!"},
        {"role": "user", "content": "What's my name?"}
    ]
)
print("Multi-message:", response3.output_text[:50])
