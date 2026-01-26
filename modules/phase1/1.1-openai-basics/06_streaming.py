"""
06 - Streaming Responses
========================
Get responses token-by-token instead of waiting for completion.

Key concept: stream=True returns an iterator, not a complete response.
"""

from openai import OpenAI

client = OpenAI()

# Non-streaming: Wait for full response
print("Non-streaming (waits for completion):")
response = client.responses.create(
    model="gpt-4o-mini",
    input="Count from 1 to 10 slowly.",
    stream=False  # Default
)
print(response.output_text)
print()

# Streaming: Get tokens as they arrive
print("Streaming (tokens arrive progressively):")
stream = client.responses.create(
    model="gpt-4o-mini",
    input="Count from 1 to 10 slowly.",
    stream=True
)

# Iterate over stream events
for event in stream:
    # Events have different types
    if event.type == "response.output_text.delta":
        # This event contains a chunk of text
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed":
        # Final event with full response
        print("\n\n[Stream completed]")
