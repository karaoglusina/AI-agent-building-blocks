"""
04 - Classification
===================
Classify text into predefined categories.

Key concept: Use Literal types to restrict classification options.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Literal
from pydantic import BaseModel
from openai import OpenAI
from utils.data_loader import load_sample_jobs

client = OpenAI()


class JobClassification(BaseModel):
    """Classification of a job posting."""
    
    # Restricted to specific values
    seniority: Literal["junior", "mid", "senior", "lead", "executive"]
    category: Literal["engineering", "data", "product", "design", "business", "other"]
    technical_level: Literal["low", "medium", "high"]
    
    # Free text for reasoning
    reasoning: str


# Classify multiple jobs
jobs = load_sample_jobs(5)

for job in jobs:
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Classify job postings accurately based on the title and description."
            },
            {
                "role": "user",
                "content": f"Classify this job:\nTitle: {job['title']}\nDescription: {job['description'][:500]}"
            }
        ],
        text_format=JobClassification,
    )
    
    result = response.output_parsed
    
    print(f"📋 {job['title']}")
    print(f"   Seniority: {result.seniority}")
    print(f"   Category: {result.category}")
    print(f"   Technical: {result.technical_level}")
    print(f"   Why: {result.reasoning[:80]}...")
    print()
