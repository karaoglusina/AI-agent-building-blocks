"""
06 - Batch Processing
=====================
Process multiple items efficiently with structured output.

Key concept: Process many items, collect structured results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel
from openai import OpenAI
from utils.data_loader import load_sample_jobs

client = OpenAI()


class QuickJobSummary(BaseModel):
    """Quick summary for batch processing."""
    title: str
    main_skill: str
    experience_years: int
    is_technical: bool


def summarize_job(job: dict) -> QuickJobSummary:
    """Summarize a single job."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=f"Summarize: {job['title']}\n{job['description'][:300]}",
        text_format=QuickJobSummary,
    )
    return response.output_parsed


# Process batch of jobs
jobs = load_sample_jobs(10)
summaries = []

print("Processing jobs...")
for i, job in enumerate(jobs):
    summary = summarize_job(job)
    summaries.append(summary)
    print(f"  [{i+1}/{len(jobs)}] {summary.title}")

# Analyze results
print("\n" + "=" * 60)
print("📊 Batch Analysis")
print("=" * 60)

# Technical vs non-technical
technical = sum(1 for s in summaries if s.is_technical)
print(f"\nTechnical roles: {technical}/{len(summaries)}")

# Experience distribution
exp_levels = {}
for s in summaries:
    level = "0-2" if s.experience_years <= 2 else "3-5" if s.experience_years <= 5 else "5+"
    exp_levels[level] = exp_levels.get(level, 0) + 1
print(f"Experience distribution: {exp_levels}")

# Top skills
from collections import Counter
skills = Counter(s.main_skill for s in summaries)
print(f"Top skills: {skills.most_common(5)}")
