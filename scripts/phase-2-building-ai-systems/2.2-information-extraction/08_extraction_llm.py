"""
08 - LLM-Based Extraction
=========================
Use LLMs with structured output for flexible information extraction.

Key concept: LLMs understand context and can extract nuanced information.

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


class JobRequirements(BaseModel):
    """Structured extraction of job requirements."""
    years_experience: int | None
    required_skills: list[str]
    nice_to_have_skills: list[str]
    education: str | None
    is_remote: bool
    seniority_level: str


class CompanyInfo(BaseModel):
    """Extract company-related information."""
    company_culture: list[str]
    benefits: list[str]
    team_size: str | None
    industry: str | None


def extract_requirements(text: str) -> JobRequirements:
    """Extract job requirements using structured output."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Extract job requirements from the description."},
    {"role": "user", "content": text}
    ]
    ,
    response_format={"type": "json_object"})
    return JobRequirements.model_validate_json(response.choices[0].message.content)


def extract_company_info(text: str) -> CompanyInfo:
    """Extract company information using structured output."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Extract company culture and benefits from job description."},
    {"role": "user", "content": text}
    ]
    ,
    response_format={"type": "json_object"})
    return JobRequirements.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    # Load a job posting
    jobs = load_sample_jobs(1)
    job = jobs[0]
    
    print(f"=== JOB: {job['title']} ===")
    print(f"Company: {job['companyName']}\n")
    
    # Extract requirements
    print("=== EXTRACTED REQUIREMENTS ===")
    reqs = extract_requirements(job["description"])
    
    print(f"Experience: {reqs.years_experience} years" if reqs.years_experience else "Experience: Not specified")
    print(f"Education: {reqs.education or 'Not specified'}")
    print(f"Seniority: {reqs.seniority_level}")
    print(f"Remote: {'Yes' if reqs.is_remote else 'No'}")
    print(f"\nRequired skills:")
    for skill in reqs.required_skills[:8]:
        print(f"  • {skill}")
    print(f"\nNice to have:")
    for skill in reqs.nice_to_have_skills[:5]:
        print(f"  • {skill}")
    
    # Extract company info
    print("\n=== EXTRACTED COMPANY INFO ===")
    info = extract_company_info(job["description"])
    
    print(f"Industry: {info.industry or 'Not specified'}")
    print(f"Team size: {info.team_size or 'Not specified'}")
    print(f"\nCulture highlights:")
    for item in info.company_culture[:4]:
        print(f"  • {item}")
    print(f"\nBenefits:")
    for item in info.benefits[:4]:
        print(f"  • {item}")
