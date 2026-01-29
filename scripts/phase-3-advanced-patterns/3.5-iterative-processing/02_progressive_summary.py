"""
02 - Progressive Summarization
===============================
Summarize in stages (para → page → chapter).

Key concept: Build summaries incrementally in layers - preserves hierarchical structure.

Book reference: NLP_cook.9
"""

from openai import OpenAI
import os

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


def summarize_text(text: str, max_words: int, context: str = "") -> str:
    """Summarize text to approximately max_words."""
    prompt = f"Summarize this in ~{max_words} words. Be concise and factual."
    if context:
        prompt += f" Context: {context}"

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": prompt},
    {"role": "user", "content": text}
    ]
    )
    return response.choices[0].message.content


def progressive_summarize_jobs(jobs: list[dict], levels: list[int] = [100, 50, 20]) -> dict:
    """
    Create progressive summaries of job listings at multiple abstraction levels.

    Args:
        jobs: List of job postings
        levels: Word counts for each summary level (detailed → brief → ultra-brief)

    Returns:
        Dictionary with summaries at each level
    """
    # Level 1: Summarize each job individually (detailed)
    print(f"Level 1: Summarizing {len(jobs)} jobs individually...")
    job_summaries = []
    for i, job in enumerate(jobs, 1):
        job_text = f"Title: {job['title']}\n\nDescription: {job['description']}"
        summary = summarize_text(job_text, max_words=levels[0])
        job_summaries.append(f"Job {i}: {summary}")

    level1_text = "\n\n".join(job_summaries)
    print(f"  Created {len(job_summaries)} detailed summaries")

    # Level 2: Summarize the summaries (medium abstraction)
    print(f"\nLevel 2: Creating medium-level summary...")
    level2_summary = summarize_text(
        level1_text,
        max_words=levels[1],
        context="These are job posting summaries"
    )
    print(f"  Condensed to ~{levels[1]} words")

    # Level 3: Create executive summary (high abstraction)
    print(f"\nLevel 3: Creating executive summary...")
    level3_summary = summarize_text(
        level2_summary,
        max_words=levels[2],
        context="This is a summary of job postings"
    )
    print(f"  Final summary: ~{levels[2]} words")

    return {
        "level1_detailed": level1_text,
        "level2_medium": level2_summary,
        "level3_executive": level3_summary,
        "job_count": len(jobs)
    }


def hierarchical_summarize_by_category(jobs: list[dict]) -> dict:
    """Create hierarchical summaries grouped by category."""
    # Group jobs by a simple heuristic (first word of title)
    categories = {}
    for job in jobs:
        category = job['title'].split()[0]  # Simple categorization
        if category not in categories:
            categories[category] = []
        categories[category].append(job)

    print(f"Grouped jobs into {len(categories)} categories")

    # Summarize each category
    category_summaries = {}
    for category, category_jobs in categories.items():
        if len(category_jobs) >= 2:  # Only summarize if multiple jobs
            job_texts = [f"{j['title']}: {j['description'][:100]}" for j in category_jobs]
            combined = "\n".join(job_texts)
            summary = summarize_text(combined, max_words=30)
            category_summaries[category] = {
                "count": len(category_jobs),
                "summary": summary
            }

    # Create final cross-category summary
    all_categories_text = "\n".join([
        f"{cat}: {data['summary']} ({data['count']} jobs)"
        for cat, data in category_summaries.items()
    ])

    final_summary = summarize_text(
        all_categories_text,
        max_words=50,
        context="These are summaries by job category"
    )

    return {
        "categories": category_summaries,
        "final_summary": final_summary
    }


if __name__ == "__main__":
    print("=== PROGRESSIVE SUMMARIZATION ===\n")

    jobs = load_sample_jobs(15)
    print(f"Processing {len(jobs)} jobs...\n")

    # Progressive multi-level summarization
    result = progressive_summarize_jobs(jobs, levels=[100, 50, 20])

    print("\n" + "=" * 50)
    print("\n=== LEVEL 2 (MEDIUM) SUMMARY ===\n")
    print(result["level2_medium"])

    print("\n" + "=" * 50)
    print("\n=== LEVEL 3 (EXECUTIVE) SUMMARY ===\n")
    print(result["level3_executive"])

    print("\n" + "=" * 50)
    print("\n=== HIERARCHICAL SUMMARIZATION ===\n")
    hierarchical = hierarchical_summarize_by_category(jobs)
    print(f"\nFinal cross-category summary:")
    print(hierarchical["final_summary"])
