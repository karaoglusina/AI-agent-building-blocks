"""
06 - KeyBERT Extraction
=======================
Embedding-based keyword extraction using KeyBERT.

Key concept: KeyBERT uses embeddings to find words most similar to the document.

Book reference: NLP_cook.5
"""

from keybert import KeyBERT

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

# Initialize KeyBERT with default model
kw_model = KeyBERT()


def extract_keywords_simple(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Extract keywords using default settings."""
    return kw_model.extract_keywords(text, top_n=top_n)


def extract_keywords_ngrams(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Extract keyword phrases (1-3 words)."""
    return kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n
    )


def extract_diverse_keywords(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    """Extract diverse keywords using MMR (Maximal Marginal Relevance)."""
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,  # Maximize diversity
        diversity=0.7,  # Higher = more diverse
        top_n=top_n
    )


if __name__ == "__main__":
    # Sample job description
    jobs = load_sample_jobs(1)
    text = jobs[0]["description"]
    title = jobs[0]["title"]
    
    print(f"=== JOB: {title} ===\n")
    
    # Simple extraction
    print("SIMPLE KEYWORDS:")
    simple = extract_keywords_simple(text, top_n=5)
    for word, score in simple:
        print(f"  {word:<30} {score:.4f}")
    print()
    
    # N-gram extraction
    print("KEYPHRASE EXTRACTION (1-3 words):")
    ngrams = extract_keywords_ngrams(text, top_n=5)
    for phrase, score in ngrams:
        print(f"  {phrase:<30} {score:.4f}")
    print()
    
    # Diverse extraction
    print("DIVERSE KEYWORDS (MMR):")
    diverse = extract_diverse_keywords(text, top_n=5)
    for word, score in diverse:
        print(f"  {word:<30} {score:.4f}")
    print()
    
    # Compare across multiple jobs
    print("=" * 50)
    print("=== COMPARING MULTIPLE JOBS ===\n")
    jobs = load_sample_jobs(3)
    for job in jobs:
        keywords = extract_diverse_keywords(job["description"], top_n=3)
        kw_str = ", ".join(w for w, s in keywords)
        print(f"📋 {job['title'][:40]}")
        print(f"   Keywords: {kw_str}\n")
