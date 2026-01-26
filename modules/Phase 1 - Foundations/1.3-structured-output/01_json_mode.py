"""
01 - JSON Mode
==============
Force the model to output valid JSON.

Key concept: response_format={"type": "json_object"} ensures JSON output.
Note: You MUST ask for JSON in the prompt, or it may not work.
"""

import json
from openai import OpenAI

client = OpenAI()

# Without JSON mode - might get markdown or plain text
response1 = client.responses.create(
    model="gpt-4o-mini",
    input="List 3 programming languages with their use cases."
)
print("Without JSON mode:")
print(response1.output_text[:200])
print()

# With JSON mode - guaranteed valid JSON
response2 = client.responses.create(
    model="gpt-4o-mini",
    input="List 3 programming languages with their use cases. Return as JSON.",
    text={"format": {"type": "json_object"}}  # Force JSON output
)
print("With JSON mode:")
print(response2.output_text)
print()

# Parse the JSON
data = json.loads(response2.output_text)
print(f"Parsed type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
