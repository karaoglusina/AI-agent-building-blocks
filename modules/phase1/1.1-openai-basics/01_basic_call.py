"""
01 - Basic API Call
====================
The simplest possible OpenAI API call using the Responses API.

Key concept: client.responses.create() is the core method.
"""

from openai import OpenAI

client = OpenAI()  # Reads OPENAI_API_KEY from environment

response = client.responses.create(
    model="gpt-4o-mini",  # Cheaper model for learning
    input="What is Python?"
)

# The simplest way to get the text output
print(response.output_text)
