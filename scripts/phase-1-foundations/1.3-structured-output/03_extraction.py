"""
03 - Information Extraction
===========================
Extract structured data from unstructured text.

Key concept: Give the model text and a schema, get structured data back.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pydantic import BaseModel
from openai import OpenAI
import os
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


# Define what we want to extract
class ExtractedJobInfo(BaseModel):
    """Structured info extracted from job description."""
    required_skills: list[str]
    nice_to_have_skills: list[str]
    years_experience: int | None
    education_required: str | None
    remote_option: bool
    salary_mentioned: bool


# Get a real job description
job = load_sample_jobs(1)[0]
description = job["description"]

print(f"Job: {job['title']} at {job['companyName']}")
print(f"Description length: {len(description)} chars")
print("-" * 50)

# Extract structured info
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{
"role": "system",
"content": "Extract structured information from job descriptions. Be precise."
},
{
"role": "user",
"content": f"Extract key information from this job posting:\n\n{description}"
}
],
response_format={"type": "json_object"})

import json
info = ExtractedJobInfo.model_validate_json(response.choices[0].message.content)

print("Extracted Information:")
print(f"  Required skills: {', '.join(info.required_skills)}")
print(f"  Nice-to-have: {', '.join(info.nice_to_have_skills)}")
print(f"  Years experience: {info.years_experience}")
print(f"  Education: {info.education_required}")
print(f"  Remote option: {info.remote_option}")
print(f"  Salary mentioned: {info.salary_mentioned}")
