"""
02 - Request/Response Validation
=================================
Use Pydantic models for automatic validation and serialization.

Key concept: Pydantic models ensure type safety and provide automatic validation for API inputs/outputs.

Book reference: AI_eng.2
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uvicorn

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

app = FastAPI(title="Pydantic Validation API")


# Request models
class JobQuery(BaseModel):
    """Query model for job search."""
    query: str = Field(min_length=3, max_length=200, description="Search query")
    location: Optional[str] = Field(None, max_length=100)
    max_results: int = Field(default=10, ge=1, le=100, description="Number of results")
    remote_only: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class CandidateProfile(BaseModel):
    """Model for candidate profile."""
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    skills: list[str] = Field(min_length=1, max_length=20)
    years_experience: int = Field(ge=0, le=50)
    remote_preference: bool = Field(default=True)


# Response models
class JobResult(BaseModel):
    """Single job result."""
    id: str
    title: str
    company: str
    location: str
    match_score: float = Field(ge=0.0, le=1.0)


class JobSearchResponse(BaseModel):
    """Response for job search."""
    query: str
    total_results: int
    results: list[JobResult]
    message: str


@app.post("/search", response_model=JobSearchResponse)
def search_jobs(query: JobQuery) -> JobSearchResponse:
    """
    Search for jobs with validated query parameters.

    FastAPI automatically validates the request body against JobQuery model.
    """
    # Mock results
    mock_results = [
        JobResult(
            id=f"job-{i}",
            title=f"Senior Python Developer {i}",
            company=f"Tech Company {i}",
            location=query.location or "Remote",
            match_score=0.95 - (i * 0.05)
        )
        for i in range(1, min(query.max_results, 4))
    ]

    return JobSearchResponse(
        query=query.query,
        total_results=len(mock_results),
        results=mock_results,
        message=f"Found {len(mock_results)} jobs matching '{query.query}'"
    )


@app.post("/candidate/register")
def register_candidate(profile: CandidateProfile) -> dict[str, str]:
    """
    Register a candidate with validated profile.

    Pydantic validates email format, skill count, experience range, etc.
    """
    return {
        "status": "success",
        "message": f"Registered {profile.name} with {len(profile.skills)} skills",
        "candidate_id": f"cand-{hash(profile.email) % 10000}"
    }


@app.get("/validate-demo")
def validation_demo() -> dict[str, str]:
    """Demo endpoint showing validation examples."""
    return {
        "message": "Try POST requests to /search or /candidate/register",
        "search_example": {
            "query": "Python developer",
            "location": "San Francisco",
            "max_results": 5,
            "remote_only": True
        },
        "candidate_example": {
            "name": "John Doe",
            "email": "john@example.com",
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "years_experience": 5,
            "remote_preference": True
        }
    }


if __name__ == "__main__":
    print("Starting Pydantic Validation API...")
    print("📚 API docs: http://localhost:8000/docs")
    print("\nTry posting to /search with:")
    print('  curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \\')
    print('    -d \'{"query": "Python developer", "max_results": 3}\'')
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
