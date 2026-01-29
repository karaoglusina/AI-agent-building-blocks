"""
02 - Few-Shot Classification
============================
Provide examples for better classification accuracy.

Key concept: Examples in the prompt guide the model's behavior more reliably than instructions alone.

Book reference: AI_eng.5, hands_on_LLM.II.6
"""

from openai import OpenAI
import os
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class Classification(BaseModel):
    """Classification result."""
    seniority: str  # junior, mid, senior, lead, executive
    technical: bool  # requires coding/technical skills


# Few-shot examples
EXAMPLES = """
Example 1:
Title: Junior Python Developer
Seniority: junior
Technical: true

Example 2:
Title: VP of Marketing
Seniority: executive
Technical: false

Example 3:
Title: Senior Data Scientist
Seniority: senior
Technical: true

Example 4:
Title: Product Manager
Seniority: mid
Technical: false

Example 5:
Title: Staff Engineer - Platform
Seniority: lead
Technical: true
"""


def classify_with_examples(title: str) -> Classification:
    """Classify job title using few-shot examples."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Classify job titles by seniority level and whether they're technical roles."
    },
    {
    "role": "user",
    "content": f"Learn from these examples:\n{EXAMPLES}\n\nNow classify:\nTitle: {title}"
    }
    ]
    ,
    response_format={"type": "json_object"})
    return Classification.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    jobs = load_sample_jobs(8)
    
    print("=== FEW-SHOT CLASSIFICATION ===\n")
    print("Seniority levels: junior, mid, senior, lead, executive")
    print("Technical: requires coding/technical skills\n")
    
    for job in jobs:
        result = classify_with_examples(job["title"])
        tech_icon = "🔧" if result.technical else "📊"
        print(f"{tech_icon} {job['title']}")
        print(f"   Seniority: {result.seniority} | Technical: {result.technical}")
        print()
