"""
05 - Sentence Segmentation
==========================
Split text into sentences using different methods.

Key concept: Sentence boundaries are tricky - "Dr. Smith" shouldn't split.

Book reference: NLP_cook.1, speach_lang.I.2.7
"""

import nltk
import spacy
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

nlp = spacy.load("en_core_web_sm")


def segment_nltk(text: str) -> list[str]:
    """Segment using NLTK's Punkt tokenizer."""
    return nltk.sent_tokenize(text)


def segment_spacy(text: str) -> list[str]:
    """Segment using spaCy's statistical model."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def segment_simple(text: str) -> list[str]:
    """Simple regex-based segmentation (for comparison)."""
    # Split on period, exclamation, question mark followed by space and capital
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)


if __name__ == "__main__":
    # Tricky text with abbreviations
    tricky_text = """Dr. Smith works at ABC Inc. in New York. 
    She has a Ph.D. in Computer Science. The role pays $100K+ per year!
    Do you have 3+ years of experience? We're looking for Sr. Engineers."""
    
    print("=== TRICKY TEXT ===")
    print(tricky_text)
    print()
    
    print("=== SIMPLE REGEX (often wrong) ===")
    simple = segment_simple(tricky_text)
    for i, sent in enumerate(simple, 1):
        print(f"  {i}. {sent.strip()}")
    print()
    
    print("=== NLTK (handles abbreviations) ===")
    nltk_sents = segment_nltk(tricky_text)
    for i, sent in enumerate(nltk_sents, 1):
        print(f"  {i}. {sent.strip()}")
    print()
    
    print("=== SPACY (best accuracy) ===")
    spacy_sents = segment_spacy(tricky_text)
    for i, sent in enumerate(spacy_sents, 1):
        print(f"  {i}. {sent.strip()}")
    print()
    
    # Real job description
    print("=== REAL JOB DESCRIPTION ===")
    jobs = load_sample_jobs(1)
    desc = jobs[0]["description"][:600]
    sentences = segment_spacy(desc)
    print(f"Found {len(sentences)} sentences")
    for sent in sentences[:3]:
        print(f"  • {sent[:80]}...")
