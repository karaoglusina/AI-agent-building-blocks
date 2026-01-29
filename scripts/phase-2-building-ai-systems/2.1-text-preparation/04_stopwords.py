"""
04 - Stopword Removal
=====================
Remove common words that don't carry meaning for analysis.

Key concept: Stopwords removal reduces noise but can lose context - use judiciously.

Book reference: NLP_cook.1
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import nltk
except ImportError:
    MISSING_DEPENDENCIES.append('nltk')

try:
    import spacy
except ImportError:
    MISSING_DEPENDENCIES.append('spacy')


# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)


nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path

# Load resources
nlp = spacy.load("en_core_web_sm")
NLTK_STOPWORDS = set(stopwords.words("english"))


def remove_stopwords_nltk(words: list[str]) -> list[str]:
    """Remove stopwords using NLTK's English list (179 words)."""
    return [w for w in words if w.lower() not in NLTK_STOPWORDS]


def remove_stopwords_spacy(text: str) -> list[str]:
    """Remove stopwords using spaCy's list (326 words)."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]


def create_custom_stopwords() -> set[str]:
    """Create domain-specific stopwords for job postings."""
    base = NLTK_STOPWORDS.copy()
    # Add job-posting specific stopwords
    job_stopwords = {
        "position", "role", "opportunity", "candidate", "team",
        "company", "experience", "work", "working", "apply",
        "job", "looking", "join", "requirements", "qualifications"
    }
    return base | job_stopwords


if __name__ == "__main__":
    # Load a job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"][:500]
    
    print("=== NLTK STOPWORDS ===")
    print(f"Total: {len(NLTK_STOPWORDS)} words")
    print(f"Sample: {list(NLTK_STOPWORDS)[:15]}")
    print()
    
    print("=== SPACY STOPWORDS ===")
    print(f"Total: {len(nlp.Defaults.stop_words)} words")
    print()
    
    # Compare removal
    words = nltk.word_tokenize(text)
    nltk_filtered = remove_stopwords_nltk(words)
    spacy_filtered = remove_stopwords_spacy(text)
    
    print("=== REMOVAL COMPARISON ===")
    print(f"Original words: {len(words)}")
    print(f"After NLTK removal: {len(nltk_filtered)}")
    print(f"After spaCy removal: {len(spacy_filtered)}")
    print()
    
    print("=== CUSTOM STOPWORDS FOR JOBS ===")
    custom = create_custom_stopwords()
    custom_filtered = [w for w in words if w.lower() not in custom]
    print(f"With custom list: {len(custom_filtered)} words")
    print(f"Key content words: {custom_filtered[:20]}")
