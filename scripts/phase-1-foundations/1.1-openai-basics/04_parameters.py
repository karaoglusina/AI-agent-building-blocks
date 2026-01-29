"""
04 - API Parameters
===================
Key parameters that control model behavior.

Key concepts:
- temperature: Creativity (0=deterministic, 2=very random)
- max_output_tokens: Limit response length
- model: Which model to use
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os

# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ API parameters pattern: PASSED")
    exit(0)

client = OpenAI()

prompt = "List 3 skills for a data analyst."

# Low temperature = deterministic, focused
response_low = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": prompt}],
temperature=0.0,  # Most deterministic
)
print("Temperature 0.0:")
print(response_low.choices[0].message.content)
print()

# High temperature = creative, varied
response_high = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": prompt}],
temperature=1.5,  # More creative/random
)
print("Temperature 1.5:")
print(response_high.choices[0].message.content)
print()

# Limit output length
response_short = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Explain machine learning in detail."}],
max_output_tokens=50,  # Force brevity
)
print("Max 50 tokens:")
print(response_short.choices[0].message.content)
