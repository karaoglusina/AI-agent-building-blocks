"""
09 - Metadata Extraction
========================
Extract structural metadata from documents (titles, dates, structure).

Key concept: Metadata enables filtering and improves retrieval relevance.

Book reference: AI_eng.8
"""

import re
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs


def extract_metadata(job: dict) -> dict:
    """Extract comprehensive metadata from a job posting."""
    description = job.get("description", "")
    
    metadata = {
        # Direct fields
        "id": job.get("id"),
        "title": job.get("title"),
        "company": job.get("companyName"),
        "location": job.get("location"),
        "sector": job.get("sector"),
        "experience_level": job.get("experienceLevel"),
        "work_type": job.get("workType"),
        
        # Derived fields
        "word_count": len(description.split()),
        "has_salary": bool(job.get("salary")),
        "posted_date": parse_date(job.get("publishedAt")),
    }
    
    # Extract from description text
    metadata.update({
        "years_experience": extract_years_experience(description),
        "degree_required": extract_education(description),
        "is_remote": is_remote_job(description, job.get("workType", "")),
        "section_count": count_sections(description),
    })
    
    return metadata


def extract_years_experience(text: str) -> int | None:
    """Extract years of experience requirement."""
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'(\d+)-\d+\s+(?:years?|yrs?)',
        r'experience:\s*(\d+)\+?\s*(?:years?|yrs?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_education(text: str) -> str | None:
    """Extract education requirements."""
    patterns = {
        "PhD": r"ph\.?d|doctorate",
        "Masters": r"master'?s?\s+degree|m\.?s\.?\s+in|mba",
        "Bachelors": r"bachelor'?s?|b\.?s\.?\s+in|b\.?a\.?\s+in|undergraduate",
    }
    for degree, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return degree
    return None


def is_remote_job(description: str, work_type: str) -> bool:
    """Determine if job is remote."""
    if "remote" in work_type.lower():
        return True
    remote_patterns = r'\b(remote|work from home|wfh|distributed|anywhere)\b'
    return bool(re.search(remote_patterns, description, re.IGNORECASE))


def count_sections(text: str) -> int:
    """Count major sections in the description."""
    section_patterns = [
        r'^#+\s+',                    # Markdown headers
        r'^[A-Z][A-Z\s]+:?\s*$',     # ALL CAPS HEADERS
        r'^(About|Requirements|Responsibilities|Qualifications|Benefits)',
    ]
    count = 0
    for line in text.split('\n'):
        for pattern in section_patterns:
            if re.match(pattern, line.strip()):
                count += 1
                break
    return count


def parse_date(date_str: str) -> str | None:
    """Parse date string to ISO format."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date().isoformat()
    except:
        return date_str[:10] if len(date_str) >= 10 else None


if __name__ == "__main__":
    jobs = load_sample_jobs(5)
    
    print("=== EXTRACTED METADATA ===\n")
    for job in jobs:
        meta = extract_metadata(job)
        print(f"📋 {meta['title']}")
        print(f"   Company: {meta['company']}")
        print(f"   Location: {meta['location']}")
        print(f"   Experience: {meta['years_experience']} years" if meta['years_experience'] else "   Experience: Not specified")
        print(f"   Education: {meta['degree_required'] or 'Not specified'}")
        print(f"   Remote: {'Yes' if meta['is_remote'] else 'No'}")
        print(f"   Word count: {meta['word_count']}")
        print()
