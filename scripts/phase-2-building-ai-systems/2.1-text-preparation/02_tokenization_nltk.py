"""
02 - Tokenization with NLTK
===========================
Word and sentence tokenization - splitting text into meaningful units.

Key concept: Tokenization is the first step in most NLP pipelines.

Book reference: NLP_cook.1, speach_lang.I.2.5
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    import nltk
except ImportError:
    MISSING_DEPENDENCIES.append('nltk')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path

# Download required data (run once)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def tokenize_words(text: str) -> list[str]:
    """Split text into words using NLTK word tokenizer."""
    return nltk.word_tokenize(text)


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sentence tokenizer."""
    return nltk.sent_tokenize(text)


if __name__ == "__main__":
    # Sample job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"][:800]  # First 800 chars
    
    print("=== ORIGINAL TEXT ===")
    print(text[:300] + "...")
    print()
    
    # Word tokenization
    words = tokenize_words(text)
    print("=== WORD TOKENS ===")
    print(f"Total words: {len(words)}")
    print(f"First 20: {words[:20]}")
    print()
    
    # Notice how punctuation becomes separate tokens
    print("=== PUNCTUATION HANDLING ===")
    sample = "We're looking for a Sr. Developer (Python/Java)."
    print(f"Input: {sample}")
    print(f"Tokens: {tokenize_words(sample)}")
    print()
    
    # Sentence tokenization
    sentences = tokenize_sentences(text)
    print("=== SENTENCES ===")
    print(f"Total sentences: {len(sentences)}")
    for i, sent in enumerate(sentences[:3], 1):
        print(f"{i}. {sent[:80]}...")
