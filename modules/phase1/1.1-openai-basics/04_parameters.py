"""
04 - API Parameters
===================
Key parameters that control model behavior.

Key concepts:
- temperature: Creativity (0=deterministic, 2=very random)
- max_output_tokens: Limit response length
- model: Which model to use
"""

from openai import OpenAI

client = OpenAI()

prompt = "List 3 skills for a data analyst."

# Low temperature = deterministic, focused
response_low = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    temperature=0.0,  # Most deterministic
)
print("Temperature 0.0:")
print(response_low.output_text)
print()

# High temperature = creative, varied
response_high = client.responses.create(
    model="gpt-4o-mini",
    input=prompt,
    temperature=1.5,  # More creative/random
)
print("Temperature 1.5:")
print(response_high.output_text)
print()

# Limit output length
response_short = client.responses.create(
    model="gpt-4o-mini",
    input="Explain machine learning in detail.",
    max_output_tokens=50,  # Force brevity
)
print("Max 50 tokens:")
print(response_short.output_text)
