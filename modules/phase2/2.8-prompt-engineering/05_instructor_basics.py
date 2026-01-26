"""
05 - Instructor Library
=======================
Model-agnostic structured output.

Key concept: Instructor patches OpenAI to guarantee Pydantic model outputs.

Book reference: AI_eng.2
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

# Patch OpenAI client with Instructor
client = instructor.from_openai(OpenAI())


class JobMatch(BaseModel):
    """A job match result."""
    title: str
    match_score: int = Field(ge=0, le=100, description="Match percentage 0-100")
    matching_skills: list[str]
    missing_skills: list[str]
    recommendation: str


class JobAnalysis(BaseModel):
    """Analysis of a job posting."""
    seniority_level: str = Field(description="junior, mid, senior, or lead")
    is_remote: bool
    estimated_salary_range: str
    top_requirements: list[str] = Field(max_length=5)
    company_culture_signals: list[str]


def analyze_job_with_instructor(job_description: str) -> JobAnalysis:
    """Analyze a job using Instructor for structured output."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Analyze this job posting and extract structured information."
            },
            {"role": "user", "content": job_description}
        ],
        response_model=JobAnalysis,
    )


def match_candidate_to_job(
    candidate_skills: list[str],
    job_description: str
) -> JobMatch:
    """Match a candidate to a job using Instructor."""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Evaluate how well the candidate matches this job."
            },
            {
                "role": "user",
                "content": f"Candidate skills: {', '.join(candidate_skills)}\n\nJob: {job_description}"
            }
        ],
        response_model=JobMatch,
    )


if __name__ == "__main__":
    print("=== INSTRUCTOR BASICS ===\n")
    
    # Load a job
    jobs = load_sample_jobs(1)
    job = jobs[0]
    
    print(f"Job: {job['title']} at {job['companyName']}\n")
    
    # Analyze job
    print("=== JOB ANALYSIS ===")
    analysis = analyze_job_with_instructor(job["description"])
    
    print(f"Seniority: {analysis.seniority_level}")
    print(f"Remote: {analysis.is_remote}")
    print(f"Salary Range: {analysis.estimated_salary_range}")
    print(f"Top Requirements: {', '.join(analysis.top_requirements)}")
    print(f"Culture Signals: {', '.join(analysis.company_culture_signals)}")
    
    # Match candidate
    print("\n=== CANDIDATE MATCH ===")
    candidate_skills = ["Python", "Django", "PostgreSQL", "Docker"]
    
    match = match_candidate_to_job(candidate_skills, job["description"])
    
    print(f"Match Score: {match.match_score}%")
    print(f"Matching Skills: {', '.join(match.matching_skills)}")
    print(f"Missing Skills: {', '.join(match.missing_skills)}")
    print(f"Recommendation: {match.recommendation}")
    
    # Show that output is proper Pydantic model
    print("\n=== MODEL VALIDATION ===")
    print(f"Type: {type(analysis)}")
    print(f"JSON: {analysis.model_dump_json()[:200]}...")
