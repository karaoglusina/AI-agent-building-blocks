"""
02 - Field Types
================
Common Python types you can use in Pydantic models.

Key concept: Use Python type hints - Pydantic enforces them.
"""

from pydantic import BaseModel
from datetime import datetime


class JobPosting(BaseModel):
    # Required fields
    title: str
    salary: int
    is_remote: bool
    
    # Optional fields (with defaults)
    description: str = ""
    years_required: int = 0
    
    # Optional that can be None
    manager_name: str | None = None
    
    # Collections
    skills: list[str] = []
    benefits: dict[str, str] = {}
    
    # Date/time
    posted_at: datetime | None = None


# Create with minimal data
job1 = JobPosting(title="Analyst", salary=50000, is_remote=True)
print("Minimal:", job1)
print()

# Create with full data
job2 = JobPosting(
    title="Engineer",
    salary=80000,
    is_remote=False,
    description="Build stuff",
    skills=["Python", "SQL"],
    benefits={"health": "Full coverage"},
    posted_at="2024-01-15T10:30:00"  # String → datetime!
)
print("Full:", job2)
print(f"Posted at type: {type(job2.posted_at)}")
