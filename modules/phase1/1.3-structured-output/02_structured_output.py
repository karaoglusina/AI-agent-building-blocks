"""
02 - Structured Output with Pydantic
====================================
Define exact output schema using Pydantic models.

Key concept: OpenAI guarantees output matches your Pydantic schema.
"""

from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


# Define the output structure
class ProgrammingLanguage(BaseModel):
    name: str
    primary_use: str
    difficulty: str


class LanguageList(BaseModel):
    languages: list[ProgrammingLanguage]


# Request structured output
response = client.responses.parse(  # Note: .parse() not .create()
    model="gpt-4o-mini",
    input="List 3 programming languages with their primary use and difficulty level.",
    text_format=LanguageList,  # Pydantic model as schema
)

# response.output_parsed is already a Pydantic model!
result: LanguageList = response.output_parsed

print(f"Type: {type(result)}")
print(f"Number of languages: {len(result.languages)}")
print()

for lang in result.languages:
    print(f"• {lang.name}")
    print(f"  Use: {lang.primary_use}")
    print(f"  Difficulty: {lang.difficulty}")
    print()
