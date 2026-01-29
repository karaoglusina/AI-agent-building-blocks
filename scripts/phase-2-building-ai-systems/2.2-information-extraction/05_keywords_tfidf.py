"""
05 - TF-IDF Keywords
====================
Extract keywords using TF-IDF (Term Frequency-Inverse Document Frequency).

Key concept: TF-IDF finds words that are important in a document but rare overall.

Book reference: NLP_cook.3, speach_lang.I.6.5
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    MISSING_DEPENDENCIES.append('sklearn')

import numpy as np
import sys
from pathlib import Path
import os

# Skip if dependencies missing in TEST_MODE
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs


def extract_tfidf_keywords(documents: list[str], top_n: int = 10) -> list[list[tuple[str, float]]]:
    """
    Extract top keywords for each document using TF-IDF.
    
    Returns list of (word, score) tuples for each document.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),  # Include bigrams
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    results = []
    for doc_idx in range(len(documents)):
        # Get scores for this document
        doc_vector = tfidf_matrix[doc_idx].toarray().flatten()
        
        # Get top N indices
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        
        keywords = [
            (feature_names[idx], doc_vector[idx])
            for idx in top_indices
            if doc_vector[idx] > 0
        ]
        results.append(keywords)
    
    return results


def compare_documents(docs: list[str], labels: list[str]) -> None:
    """Compare keywords across multiple documents."""
    all_keywords = extract_tfidf_keywords(docs, top_n=5)
    
    for label, keywords in zip(labels, all_keywords):
        print(f"\n{label}:")
        for word, score in keywords:
            bar = "█" * int(score * 20)
            print(f"  {word:<25} {score:.3f} {bar}")


if __name__ == "__main__":
    # Load job descriptions
    jobs = load_sample_jobs(5)
    descriptions = [job["description"] for job in jobs]
    titles = [job["title"] for job in jobs]
    
    print("=== TF-IDF KEYWORDS PER JOB ===")
    keywords_per_job = extract_tfidf_keywords(descriptions, top_n=8)
    
    for title, keywords in zip(titles, keywords_per_job):
        print(f"\n📋 {title}")
        for word, score in keywords[:5]:
            print(f"   • {word} ({score:.3f})")
    
    # Compare two specific documents
    print("\n" + "=" * 50)
    print("=== COMPARING TWO JOBS ===")
    if len(descriptions) >= 2:
        compare_documents(
            descriptions[:2],
            [f"Job 1: {titles[0][:30]}", f"Job 2: {titles[1][:30]}"]
        )
    
    # Show how TF-IDF differs from raw frequency
    print("\n" + "=" * 50)
    print("=== TF-IDF vs RAW FREQUENCY ===")
    print("TF-IDF downweights common words like 'experience', 'team', 'work'")
    print("and highlights distinctive terms for each job posting.")
