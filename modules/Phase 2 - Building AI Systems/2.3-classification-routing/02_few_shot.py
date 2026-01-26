"""
02 - Few-Shot Classification
============================
Provide examples for better classification accuracy.

Key concept: Examples in the prompt guide the model's behavior more reliably than instructions alone.

Book reference: AI_eng.5, hands_on_LLM.II.6
"""

from openai import OpenAI
from pydantic import BaseModel

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

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
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Classify job titles by seniority level and whether they're technical roles."
            },
            {
                "role": "user",
                "content": f"Learn from these examples:\n{EXAMPLES}\n\nNow classify:\nTitle: {title}"
            }
        ],
        text_format=Classification
    )
    return response.output_parsed


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
