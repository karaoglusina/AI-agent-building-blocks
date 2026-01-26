"""
Data loading utilities for job post data.
"""

import json
from pathlib import Path

# Path to data file (relative to project root)
DATA_PATH = Path(__file__).parent.parent / "data" / "job_post_data.json"


def load_jobs() -> list[dict]:
    """Load all job posts from JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_jobs(n: int = 10) -> list[dict]:
    """Load first n job posts for quick testing."""
    return load_jobs()[:n]


def get_job_by_id(job_id: str) -> dict | None:
    """Get a specific job by ID."""
    for job in load_jobs():
        if job.get("id") == job_id:
            return job
    return None


if __name__ == "__main__":
    # Quick test
    jobs = load_sample_jobs(3)
    print(f"Loaded {len(jobs)} jobs")
    print(f"First job title: {jobs[0]['title']}")
