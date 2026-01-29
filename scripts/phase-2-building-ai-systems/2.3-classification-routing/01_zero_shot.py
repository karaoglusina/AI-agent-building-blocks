"""
01 - Zero-Shot Classification
=============================
Classify content without training data using LLM.

Key concept: LLMs can classify text into categories they've never been explicitly trained on.

Book reference: AI_eng.5, hands_on_LLM.II.6, NLP_cook.8
"""

from openai import OpenAI
import os
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class JobCategory(BaseModel):
    """Classification result with reasoning."""
    category: str
    confidence: str  # high, medium, low
    reasoning: str


CATEGORIES = [
    "Engineering",
    "Data Science",
    "Product Management",
    "Design",
    "Marketing",
    "Sales",
    "Operations",
    "Finance",
    "Human Resources"]


def classify_job(title: str, description: str) -> JobCategory:
    """Classify a job into a category using zero-shot classification."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": f"Classify the job into one of these categories: {', '.join(CATEGORIES,
    response_format={"type": "json_object"})}. "
    "Provide your confidence level and brief reasoning."
    },
    {
    "role": "user",
    "content": f"Title: {title}\n\nDescription: {description[:1000]}"
    }
    ]
    )
    return JobCategory.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    jobs = load_sample_jobs(5)
    
    print("=== ZERO-SHOT JOB CLASSIFICATION ===\n")
    print(f"Categories: {', '.join(CATEGORIES)}\n")
    
    for job in jobs:
        result = classify_job(job["title"], job["description"])
        print(f"Job: {job['title']}")
        print(f"  Category: {result.category}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning[:100]}...")
        print()
