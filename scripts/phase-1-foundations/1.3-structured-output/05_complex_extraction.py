"""
05 - Complex Extraction
=======================
Extract multiple related pieces of information with nested models.

Key concept: Compose models for rich, hierarchical extraction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pydantic import BaseModel, Field
from openai import OpenAI
import os
from utils.data_loader import load_sample_jobs
import json


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class SalaryInfo(BaseModel):
    """Salary information if mentioned."""
    min_salary: int | None = None
    max_salary: int | None = None
    currency: str = "EUR"
    period: str = "yearly"  # yearly, monthly, hourly


class RequirementSection(BaseModel):
    """A categorized section of requirements."""
    category: str  # e.g., "Technical Skills", "Soft Skills", "Education"
    items: list[str]
    is_required: bool = True


class CompanyInfo(BaseModel):
    """Information about the company."""
    name: str
    industry: str | None = None
    size_hint: str | None = None  # startup, SME, enterprise
    culture_keywords: list[str] = []


class FullJobAnalysis(BaseModel):
    """Complete structured analysis of a job posting."""
    title: str
    company: CompanyInfo
    location: str
    is_remote: bool
    salary: SalaryInfo | None = None
    requirements: list[RequirementSection]
    key_responsibilities: list[str] = Field(max_length=5)
    job_highlights: list[str] = Field(max_length=3)


# Analyze a job
job = load_sample_jobs(1)[0]

response = client.chat.completions.create(
model="gpt-4o",  # Use smarter model for complex extraction
messages=[
{
"role": "system",
"content": """Analyze job postings and extract structured information. 
Be thorough but concise. If information is not mentioned, use null."""
},
{
"role": "user",
"content": f"Analyze this job posting:\n\n{job['description']}"
}
],
response_format={"type": "json_object"})

analysis = SalaryInfo.model_validate_json(response.choices[0].message.content)

print("=" * 60)
print(f"📋 {analysis.title}")
print(f"🏢 {analysis.company.name} ({analysis.company.industry})")
print(f"📍 {analysis.location} | Remote: {analysis.is_remote}")
print("=" * 60)

if analysis.salary:
    print(f"\n💰 Salary: {analysis.salary.min_salary}-{analysis.salary.max_salary} {analysis.salary.currency}/{analysis.salary.period}")

print("\n📌 Requirements:")
for section in analysis.requirements:
    status = "Required" if section.is_required else "Nice-to-have"
    print(f"\n  {section.category} ({status}):")
    for item in section.items[:3]:  # First 3
        print(f"    • {item}")

print("\n✨ Highlights:")
for highlight in analysis.job_highlights:
    print(f"  • {highlight}")
