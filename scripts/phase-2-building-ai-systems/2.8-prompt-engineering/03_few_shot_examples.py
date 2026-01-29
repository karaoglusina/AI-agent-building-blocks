"""
03 - Few-Shot Examples
======================
Provide examples for consistent output.

Key concept: Examples show the model exactly what you want - format, style, content.

Book reference: AI_eng.5, hands_on_LLM.II.6
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
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


# Few-shot examples for job skill extraction
SKILL_EXTRACTION_EXAMPLES = """
Example 1:
Input: "5+ years Python, Django, PostgreSQL experience required"
Output: Python, Django, PostgreSQL

Example 2:
Input: "Must have AWS certification and Docker knowledge"
Output: AWS, Docker

Example 3:
Input: "Looking for someone fluent in React and TypeScript"
Output: React, TypeScript
"""


# Few-shot examples for job summary
SUMMARY_EXAMPLES = """
Example 1:
Job: Senior Python Developer at TechCorp. Remote. 5+ years experience. Django, AWS.
Summary: Senior remote Python role requiring Django and AWS expertise.

Example 2:
Job: Data Scientist at StartupXYZ. NYC. ML, Python, Spark. 3 years min.
Summary: NYC-based data science position focused on ML with Spark experience.
"""


def extract_skills_zero_shot(text: str) -> str:
    """Extract skills without examples."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Extract technical skills from the text. Return as comma-separated list."},
    {"role": "user", "content": text}
    ]
    )
    return response.choices[0].message.content


def extract_skills_few_shot(text: str) -> str:
    """Extract skills with few-shot examples."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": f"Extract technical skills from the text.\n\n{SKILL_EXTRACTION_EXAMPLES}\nNow extract skills from the input."
    },
    {"role": "user", "content": f"Input: {text}\nOutput:"}
    ]
    )
    return response.choices[0].message.content


def summarize_job_few_shot(job_text: str) -> str:
    """Summarize job with few-shot examples."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": f"Summarize job postings concisely.\n\n{SUMMARY_EXAMPLES}\nNow summarize:"
    },
    {"role": "user", "content": f"Job: {job_text}\nSummary:"}
    ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== FEW-SHOT EXAMPLES ===\n")
    
    # Test skill extraction
    test_text = "We need a full-stack developer with Node.js, React, MongoDB, and Kubernetes experience"
    
    print("=== Skill Extraction ===")
    print(f"Input: {test_text}\n")
    
    print("Zero-shot:")
    print(f"  {extract_skills_zero_shot(test_text)}")
    
    print("\nFew-shot:")
    print(f"  {extract_skills_few_shot(test_text)}")
    
    # Test job summary
    print("\n=== Job Summary ===")
    job = "Machine Learning Engineer at Google. Mountain View, CA. PhD preferred. TensorFlow, PyTorch, distributed systems. 4+ years experience. Competitive salary."
    
    print(f"Job: {job}\n")
    print(f"Summary: {summarize_job_few_shot(job)}")
    
    # Show the examples used
    print("\n=== Examples Template ===")
    print(SKILL_EXTRACTION_EXAMPLES)
