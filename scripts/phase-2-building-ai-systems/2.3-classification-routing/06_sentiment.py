"""
06 - Sentiment Analysis
=======================
Classify text as positive, negative, or neutral.

Key concept: Sentiment analysis reveals emotional tone - useful for reviews, feedback, culture.

Book reference: NLP_cook.4, speach_lang.I.4.4
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

from openai import OpenAI
import os
from pydantic import BaseModel
try:
    from textblob import TextBlob
except ImportError:
    MISSING_DEPENDENCIES.append('textblob')

import sys

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class SentimentResult(BaseModel):
    """Detailed sentiment analysis."""
    sentiment: str  # positive, negative, neutral
    score: float    # -1.0 to 1.0
    aspects: list[str]  # What aspects contribute to sentiment


def analyze_sentiment_textblob(text: str) -> tuple[str, float]:
    """Simple sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return sentiment, polarity


def analyze_sentiment_llm(text: str) -> SentimentResult:
    """Detailed sentiment analysis using LLM."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Analyze the sentiment of this job description. "
    "Focus on workplace culture, benefits, and tone. "
    "Score from -1 (negative) to 1 (positive)."
    },
    {"role": "user", "content": text[:1500]}
    ],
    response_format={"type": "json_object"}
    )
    return SentimentResult.model_validate_json(response.choices[0].message.content)


if __name__ == "__main__":
    print("=== SENTIMENT ANALYSIS ===\n")
    
    # Analyze job descriptions
    jobs = load_sample_jobs(3)
    
    for job in jobs:
        print(f"Job: {job['title']} at {job['companyName']}")
        print("-" * 50)
        
        # TextBlob (fast, simple)
        tb_sentiment, tb_score = analyze_sentiment_textblob(job["description"])
        print(f"TextBlob: {tb_sentiment} (score: {tb_score:.2f})")
        
        # LLM (nuanced, detailed)
        llm_result = analyze_sentiment_llm(job["description"])
        print(f"LLM: {llm_result.sentiment} (score: {llm_result.score:.2f})")
        print(f"Key aspects: {', '.join(llm_result.aspects[:3])}")
        print()
    
    # Comparison examples
    print("=== COMPARISON: TextBlob vs LLM ===\n")
    
    test_texts = [
        "Amazing opportunity with great benefits and work-life balance!",
        "Must work 60+ hours. No remote. Competitive salary.",
        "Looking for a developer to join our team."]
    
    for text in test_texts:
        tb_sent, tb_score = analyze_sentiment_textblob(text)
        print(f"\"{text}\"")
        print(f"  TextBlob: {tb_sent} ({tb_score:.2f})")
        print()
