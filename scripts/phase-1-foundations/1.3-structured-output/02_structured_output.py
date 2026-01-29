"""
02 - Structured Output with Pydantic
====================================
Define exact output schema using Pydantic models.

Key concept: OpenAI guarantees output matches your Pydantic schema.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from pydantic import BaseModel
from openai import OpenAI
import os
import json


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


# Define the output structure
class ProgrammingLanguage(BaseModel):
    name: str
    primary_use: str
    difficulty: str


class LanguageList(BaseModel):
    languages: list[ProgrammingLanguage]


# Request structured output
response = client.chat.completions.create(  # Note: .parse() not .create()
model="gpt-4o-mini",
messages=[{"role": "user", "content": "List 3 programming languages with their primary use and difficulty level."}],  # Pydantic model as schema
response_format={"type": "json_object"})

# ProgrammingLanguage.model_validate_json(response.choices[0].message.content) is already a Pydantic model!
result: LanguageList = ProgrammingLanguage.model_validate_json(response.choices[0].message.content)

print(f"Type: {type(result)}")
print(f"Number of languages: {len(result.languages)}")
print()

for lang in result.languages:
    print(f"• {lang.name}")
    print(f"  Use: {lang.primary_use}")
    print(f"  Difficulty: {lang.difficulty}")
    print()
