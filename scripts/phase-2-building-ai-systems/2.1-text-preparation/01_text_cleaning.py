"""
01 - Basic Text Cleaning
========================
Remove HTML, normalize whitespace, handle Unicode/encoding issues.

Key concept: Clean text before any NLP processing - garbage in, garbage out.

Book reference: AI_eng.8
"""

import re
import html
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs


def clean_text(text: str) -> str:
    """Apply standard text cleaning operations."""
    # 1. Unescape HTML entities (&amp; -> &, &lt; -> <)
    text = html.unescape(text)
    
    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Normalize Unicode (common issues: curly quotes, em-dashes)
    replacements = {
        "\u2018": "'", "\u2019": "'",  # Curly single quotes
        "\u201c": '"', "\u201d": '"',  # Curly double quotes
        "\u2014": "-", "\u2013": "-",  # Em/en dashes
        "\u2026": "...",               # Ellipsis
        "\xa0": " ",                   # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 4. Normalize whitespace (multiple spaces, tabs, newlines -> single space)
    text = re.sub(r"\s+", " ", text)
    
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    return text


if __name__ == "__main__":
    # Load a job with HTML description
    jobs = load_sample_jobs(1)
    raw_text = jobs[0].get("descriptionHtml", jobs[0]["description"])
    
    print("=== BEFORE CLEANING ===")
    print(raw_text[:500])
    print("\n" + "=" * 50 + "\n")
    
    cleaned = clean_text(raw_text)
    
    print("=== AFTER CLEANING ===")
    print(cleaned[:500])
    print(f"\n\nOriginal length: {len(raw_text)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
