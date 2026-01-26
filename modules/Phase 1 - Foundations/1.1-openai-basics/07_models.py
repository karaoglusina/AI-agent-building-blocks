"""
07 - Available Models
=====================
Different models for different use cases.

Key concept: Choose model based on task complexity and cost.
"""

from openai import OpenAI

client = OpenAI()

prompt = "What is 2+2? Answer with just the number."

# Model comparison
models = [
    ("gpt-4o-mini", "Fast, cheap, good for simple tasks"),
    ("gpt-4o", "Smartest, best for complex reasoning"),
    # ("o1-mini", "Reasoning model, thinks step-by-step"),  # Requires different params
]

for model_name, description in models:
    print(f"Model: {model_name}")
    print(f"Description: {description}")
    
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    
    print(f"Response: {response.output_text}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("-" * 40)

# Tip: For learning/testing, use gpt-4o-mini (cheapest)
# For production/complex tasks, use gpt-4o
