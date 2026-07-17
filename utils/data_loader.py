"""
Data loading utilities for job post data.
"""

import json
from pathlib import Path

# Path to data file (relative to project root)
DATA_PATH = Path(__file__).parent.parent / "data" / "sample_job_data.json"


def load_jobs(deduplicate: bool = True) -> list[dict]:
    """Load job posts from JSON file.

    The raw scrape contains the same posting more than once when it matched several
    search queries — 1,318 records, 949 unique ids, identical in every field except
    `search_params`. Duplicate ids break anything that treats the id as a key (adding
    to a vector store, upserting to a database), so the default is to drop repeats and
    keep the first occurrence.

    Pass deduplicate=False if you specifically want the raw records.
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    if not deduplicate:
        return jobs

    seen: set = set()
    unique = []
    for job in jobs:
        job_id = job.get("id")
        if job_id in seen:
            continue
        seen.add(job_id)
        unique.append(job)
    return unique


def load_sample_jobs(n: int = 10) -> list[dict]:
    """Load first n job posts for quick testing. Ids are unique."""
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
    print(f"Unique jobs available: {len(load_jobs())} (raw records: {len(load_jobs(deduplicate=False))})")
