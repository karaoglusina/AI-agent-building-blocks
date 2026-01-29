"""
03 - Part-of-Speech Tagging
===========================
Tag words with grammatical categories (NOUN, VERB, ADJ, etc.)

Key concept: POS tags reveal word roles - nouns are things, verbs are actions.

Book reference: NLP_cook.1, speach_lang.III.17.2
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import spacy
except ImportError:
    MISSING_DEPENDENCIES.append('spacy')

from collections import Counter
import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path

nlp = spacy.load("en_core_web_sm")


def get_pos_tags(text: str) -> list[tuple[str, str, str]]:
    """Get POS tags for each token in text."""
    doc = nlp(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]


def extract_by_pos(text: str, pos_types: list[str]) -> list[str]:
    """Extract words matching specific POS types."""
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in pos_types]


def get_pos_distribution(text: str) -> dict[str, int]:
    """Get distribution of POS tags in text."""
    doc = nlp(text)
    return dict(Counter(token.pos_ for token in doc))


if __name__ == "__main__":
    sample = "The senior developer built robust Python applications quickly."
    
    print("=== POS TAGS EXPLAINED ===")
    tags = get_pos_tags(sample)
    print(f"Sentence: {sample}\n")
    print(f"{'Word':<15} {'POS':<8} {'Fine-grained':<8} {'Description'}")
    print("-" * 55)
    for word, pos, tag in tags:
        desc = spacy.explain(tag) or ""
        print(f"{word:<15} {pos:<8} {tag:<8} {desc[:30]}")
    print()
    
    # Extract specific parts of speech
    print("=== EXTRACT BY POS ===")
    job_text = """
    We are looking for an experienced Python developer to build 
    scalable web applications. The ideal candidate has strong 
    analytical skills and excellent communication abilities.
    """
    
    nouns = extract_by_pos(job_text, ["NOUN", "PROPN"])
    verbs = extract_by_pos(job_text, ["VERB"])
    adjs = extract_by_pos(job_text, ["ADJ"])
    
    print(f"Nouns: {', '.join(nouns)}")
    print(f"Verbs: {', '.join(verbs)}")
    print(f"Adjectives: {', '.join(adjs)}")
    print()
    
    # Analyze job description
    print("=== JOB DESCRIPTION ANALYSIS ===")
    jobs = load_sample_jobs(1)
    dist = get_pos_distribution(jobs[0]["description"])
    
    print("POS distribution:")
    for pos, count in sorted(dist.items(), key=lambda x: -x[1])[:8]:
        desc = spacy.explain(pos) or pos
        print(f"  {pos:<6} ({desc:<15}): {count}")
