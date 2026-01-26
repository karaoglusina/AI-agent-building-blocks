"""
06 - From Raw Data
==================
Parse your job posting data into Pydantic models.

Key concept: Real-world data often needs flexible parsing.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel
from utils.data_loader import load_sample_jobs


class JobPost(BaseModel):
    """Simplified job model - only fields we care about."""
    id: str
    title: str
    companyName: str
    location: str
    description: str
    
    # Optional fields with defaults
    salary: str = ""
    experienceLevel: str = ""
    sector: str = ""

    class Config:
        extra = "ignore"  # Ignore extra fields in source data


# Load raw JSON data
raw_jobs = load_sample_jobs(5)
print(f"Loaded {len(raw_jobs)} raw jobs")
print()

# Parse into Pydantic models
jobs = [JobPost.model_validate(job) for job in raw_jobs]

# Now we have validated, typed objects!
for job in jobs:
    print(f"✓ {job.title}")
    print(f"  Company: {job.companyName}")
    print(f"  Location: {job.location}")
    print(f"  Description: {job.description[:80]}...")
    print()
