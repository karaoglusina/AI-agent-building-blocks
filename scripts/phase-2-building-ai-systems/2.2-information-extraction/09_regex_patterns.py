"""
09 - Regex Pattern Matching
===========================
Extract emails, phones, URLs, and custom patterns with regular expressions.

Key concept: Regex is fast and precise for well-defined patterns.

Book reference: NLP_cook.5, speach_lang.I.2.1
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs


# Common extraction patterns
PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
    "phone": r'\b(?:\+1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    "salary": r'\$[\d]+(?:k|K)?(?:\s*-\s*\$?[\d]+(?:k|K)?)?(?:\s*(?:per|/)\s*(?:year|yr|annum|hour|hr))?',
    "years_exp": r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
    "degree": r"(?:bachelor'?s?|master'?s?|phd|doctorate|b\.?s\.?|m\.?s\.?|mba)",
    "percentage": r'\b\d{1,3}%\b',
}


def extract_pattern(text: str, pattern_name: str) -> list[str]:
    """Extract matches for a named pattern."""
    pattern = PATTERNS.get(pattern_name, pattern_name)
    return re.findall(pattern, text, re.IGNORECASE)


def extract_all_patterns(text: str) -> dict[str, list[str]]:
    """Extract all defined patterns from text."""
    results = {}
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            results[name] = matches
    return results


def extract_skills_with_context(text: str) -> list[dict]:
    """Extract skills mentioned with experience requirements."""
    # Pattern: "5+ years of Python" or "experience with AWS"
    pattern = r'(\d+\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience\s+(?:with|in)\s+)?)?(\b[A-Z][a-z]+(?:\.[a-z]+)?)\s+(?:experience|development|programming)'
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    results = []
    for match in matches:
        years = match.group(1).strip() if match.group(1) else None
        skill = match.group(2)
        results.append({"skill": skill, "years": years})
    
    return results


def extract_tech_stack(text: str) -> list[str]:
    """Extract technology names using common patterns."""
    # Technologies often appear in specific patterns
    patterns = [
        r'(?:using|with|in)\s+([A-Z][a-zA-Z+#]+(?:\s*[/,&]\s*[A-Z][a-zA-Z+#]+)*)',
        r'(?:tech\s*stack|technologies):\s*([^.]+)']
    
    techs = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split on common separators
            items = re.split(r'[,/&]|\band\b', match)
            techs.extend(item.strip() for item in items if item.strip())
    
    return list(set(techs))


if __name__ == "__main__":
    # Sample text with various patterns
    sample = """
    Contact us at jobs@company.com or call 555-123-4567.
    Visit https://company.com/careers for more.
    Salary: $120,000 - $150,000 per year
    Requires 5+ years of experience and a Bachelor's degree.
    """
    
    print("=== SAMPLE TEXT EXTRACTION ===")
    print(sample.strip())
    print("\nExtracted:")
    results = extract_all_patterns(sample)
    for name, matches in results.items():
        print(f"  {name}: {matches}")
    print()
    
    # Job posting extraction
    print("=== JOB POSTING EXTRACTION ===")
    jobs = load_sample_jobs(2)
    
    for job in jobs:
        print(f"\n📋 {job['title']}")
        text = job["description"]
        
        # Years of experience
        years = extract_pattern(text, "years_exp")
        if years:
            print(f"   Experience required: {years[0]} years")
        
        # Degree requirements
        degrees = extract_pattern(text, "degree")
        if degrees:
            print(f"   Education: {', '.join(set(degrees))}")
        
        # Salary if mentioned
        salary = extract_pattern(text, "salary")
        if salary:
            print(f"   Salary: {salary[0]}")
        
        # URLs
        urls = extract_pattern(text, "url")
        if urls:
            print(f"   Links: {len(urls)} found")
