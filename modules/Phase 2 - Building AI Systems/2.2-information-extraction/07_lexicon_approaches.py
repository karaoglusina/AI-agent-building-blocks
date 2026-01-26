"""
07 - Lexicon & Dictionary Methods
=================================
Use word lists, dictionaries for extraction (sentiment, domain terms).

Key concept: Lexicons provide interpretable, domain-specific extraction.

Book reference: speach_lang.III.22, NLP_cook.5
"""

from textblob import TextBlob
import nltk

nltk.download("opinion_lexicon", quiet=True)
from nltk.corpus import opinion_lexicon

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs


# Custom domain lexicons
TECH_SKILLS = {
    "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
    "react", "angular", "node", "django", "flask", "fastapi", "postgresql",
    "mongodb", "redis", "git", "linux", "tensorflow", "pytorch", "pandas"
}

SOFT_SKILLS = {
    "communication", "leadership", "teamwork", "problem-solving", "analytical",
    "creative", "organized", "detail-oriented", "collaborative", "adaptable",
    "motivated", "proactive", "independent", "flexible", "reliable"
}

SENIORITY_TERMS = {
    "senior": 3, "lead": 4, "principal": 5, "staff": 4, "junior": 1,
    "entry": 1, "mid": 2, "experienced": 3, "expert": 4
}


def lexicon_sentiment(text: str) -> dict:
    """Calculate sentiment using NLTK's opinion lexicon."""
    words = text.lower().split()
    positive = set(opinion_lexicon.positive())
    negative = set(opinion_lexicon.negative())
    
    pos_words = [w for w in words if w in positive]
    neg_words = [w for w in words if w in negative]
    
    return {
        "positive_count": len(pos_words),
        "negative_count": len(neg_words),
        "positive_words": pos_words[:5],
        "negative_words": neg_words[:5],
        "sentiment_ratio": len(pos_words) / max(len(neg_words), 1)
    }


def textblob_sentiment(text: str) -> dict:
    """Get sentiment using TextBlob's built-in lexicon."""
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,      # -1 to 1
        "subjectivity": blob.sentiment.subjectivity  # 0 to 1
    }


def extract_by_lexicon(text: str, lexicon: set) -> list[str]:
    """Extract words matching a custom lexicon."""
    words = set(text.lower().split())
    # Also check for multi-word terms
    text_lower = text.lower()
    found = [term for term in lexicon if term in text_lower]
    return found


def estimate_seniority(text: str) -> tuple[str, int]:
    """Estimate job seniority level from text."""
    text_lower = text.lower()
    for term, level in sorted(SENIORITY_TERMS.items(), key=lambda x: -x[1]):
        if term in text_lower:
            return term, level
    return "unspecified", 0


if __name__ == "__main__":
    jobs = load_sample_jobs(3)
    
    print("=== LEXICON-BASED EXTRACTION ===\n")
    
    for job in jobs:
        text = job["description"]
        title = job["title"]
        
        print(f"📋 {title}")
        
        # Technical skills
        tech = extract_by_lexicon(text, TECH_SKILLS)
        print(f"   Tech skills: {', '.join(tech) if tech else 'None found'}")
        
        # Soft skills
        soft = extract_by_lexicon(text, SOFT_SKILLS)
        print(f"   Soft skills: {', '.join(soft) if soft else 'None found'}")
        
        # Seniority
        term, level = estimate_seniority(title + " " + text)
        print(f"   Seniority: {term} (level {level}/5)")
        
        # Sentiment
        sent = textblob_sentiment(text)
        print(f"   Tone: {'Positive' if sent['polarity'] > 0.1 else 'Neutral'}")
        print()
    
    print("=== SENTIMENT ANALYSIS ===")
    sample = "Great opportunity to work with amazing team on exciting projects!"
    sent = lexicon_sentiment(sample)
    print(f"Text: {sample}")
    print(f"Positive words: {sent['positive_words']}")
    print(f"Negative words: {sent['negative_words']}")
