"""
05 - Response Object
====================
Understanding what the API returns.

Key concept: The response object contains metadata, not just text.
"""

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Say hello in 3 languages."
)

# Quick access to text
print("output_text:", response.output_text)
print()

# Full response structure
print("Response ID:", response.id)
print("Model used:", response.model)
print("Created at:", response.created_at)
print()

# Output is a list (can contain multiple items)
print("Output type:", type(response.output))
print("Output length:", len(response.output))
print()

# Each output item has a type
for i, item in enumerate(response.output):
    print(f"Output[{i}] type: {item.type}")
    if item.type == "message":
        print(f"  Role: {item.role}")
        print(f"  Content: {item.content[:100]}...")

# Usage statistics
print()
print("Usage:")
print(f"  Input tokens: {response.usage.input_tokens}")
print(f"  Output tokens: {response.usage.output_tokens}")
print(f"  Total tokens: {response.usage.total_tokens}")
