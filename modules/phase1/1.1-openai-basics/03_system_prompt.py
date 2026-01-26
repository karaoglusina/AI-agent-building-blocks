"""
03 - System Prompts
===================
System prompts define the AI's behavior, personality, and constraints.

Key concept: "system" role sets context that persists through conversation.
"""

from openai import OpenAI

client = OpenAI()

# Without system prompt - generic response
response1 = client.responses.create(
    model="gpt-4o-mini",
    input="Analyze this job market trend."
)
print("Without system prompt:")
print(response1.output_text[:200])
print()

# With system prompt - focused response
response2 = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": "You are a job market analyst specializing in tech roles in the Netherlands. Be concise and data-driven."
        },
        {
            "role": "user", 
            "content": "Analyze this job market trend."
        }
    ]
)
print("With system prompt:")
print(response2.output_text[:200])
