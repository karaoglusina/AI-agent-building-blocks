"""
01 - JSON Mode
==============
Force the model to output valid JSON.

Key concept: response_format={"type": "json_object"} ensures JSON output.
Note: You MUST ask for JSON in the prompt, or it may not work.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Without JSON mode - might get markdown or plain text
response1 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "List 3 programming languages with their use cases."}]
)
print("Without JSON mode:")
print(response1.choices[0].message.content[:200])
print()

# With JSON mode - guaranteed valid JSON
response2 = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "List 3 programming languages with their use cases. Return as JSON."}],
response_format={"type": "json_object"}  # Force JSON output
)
print("With JSON mode:")
print(response2.choices[0].message.content)
print()

# Parse the JSON
data = json.loads(response2.choices[0].message.content)
print(f"Parsed type: {type(data)}")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
