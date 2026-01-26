"""
04 - Hierarchical Processing
=============================
Process content at multiple levels of abstraction.

Key concept: Work from fine to coarse (or reverse) - extract details first, then combine into higher-level insights.

Book reference: AI_eng.6
"""

from openai import OpenAI
from typing import Any
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


def extract_job_details(job: dict[str, Any]) -> dict[str, Any]:
    """Level 1: Extract fine-grained details from single job."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract key details from job posting. Be specific and factual."
            },
            {
                "role": "user",
                "content": f"""Job: {job['title']}
Company: {job.get('company', 'Unknown')}
Location: {job.get('location', 'Not specified')}
Description: {job.get('description', '')[:800]}

Extract:
1. Required skills (list top 5)
2. Experience level
3. Key responsibilities (list top 3)
4. Benefits mentioned"""
            }
        ],
        temperature=0.3
    )

    return {
        "job_id": job.get("id"),
        "title": job["title"],
        "details": response.choices[0].message.content
    }


def summarize_job_group(job_details: list[dict]) -> str:
    """Level 2: Summarize patterns across a group of jobs."""
    combined_details = "\n\n".join([
        f"Job: {jd['title']}\n{jd['details']}"
        for jd in job_details
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Identify common patterns and themes across multiple job postings."
            },
            {
                "role": "user",
                "content": f"""Analyze these job details and identify:
1. Most common skills required
2. Typical experience level
3. Common responsibilities
4. Typical benefits

Job details:
{combined_details[:3000]}"""
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def create_market_overview(group_summaries: list[str]) -> str:
    """Level 3: Create high-level market overview from group summaries."""
    combined_summaries = "\n\n".join([
        f"Group {i+1} Summary:\n{summary}"
        for i, summary in enumerate(group_summaries)
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Create executive summary of job market trends."
            },
            {
                "role": "user",
                "content": f"""Based on these group summaries, create a market overview covering:
1. Overall skill demand trends
2. Experience level distribution
3. Key responsibilities across roles
4. Benefits landscape

Summaries:
{combined_summaries[:3000]}"""
            }
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


def hierarchical_processing(jobs: list[dict], group_size: int = 3) -> dict:
    """Process jobs hierarchically: Job → Group → Market levels."""
    print("=== LEVEL 1: Individual Job Details ===")
    job_details = []
    for i, job in enumerate(jobs[:9]):  # Process 9 jobs
        print(f"Extracting details for job {i+1}/{9}...")
        details = extract_job_details(job)
        job_details.append(details)

    print(f"\n=== LEVEL 2: Group Summaries ===")
    group_summaries = []
    for i in range(0, len(job_details), group_size):
        group = job_details[i:i+group_size]
        print(f"Summarizing group {i//group_size + 1}...")
        summary = summarize_job_group(group)
        group_summaries.append(summary)
        print(f"Group summary: {summary[:150]}...\n")

    print("=== LEVEL 3: Market Overview ===")
    market_overview = create_market_overview(group_summaries)

    return {
        "level_1_jobs": len(job_details),
        "level_2_groups": len(group_summaries),
        "level_3_overview": market_overview
    }


if __name__ == "__main__":
    # Load sample jobs
    jobs = load_sample_jobs(10)

    print("Starting hierarchical processing...")
    print("This will process jobs at 3 levels of abstraction:\n")

    result = hierarchical_processing(jobs, group_size=3)

    print("\n" + "=" * 70)
    print("=== FINAL MARKET OVERVIEW ===")
    print(result["level_3_overview"])

    print(f"\n\nProcessed {result['level_1_jobs']} individual jobs")
    print(f"Created {result['level_2_groups']} group summaries")
    print("Generated 1 market overview")
    print("\nHierarchical processing complete!")
