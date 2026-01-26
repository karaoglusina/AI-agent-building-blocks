"""
06 - Constrained Generation
===========================
Force output to follow patterns.

Key concept: Constraints ensure output matches expected formats.

Book reference: hands_on_LLM.II.6 (Grammar)
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum

client = OpenAI()


class Sentiment(str, Enum):
    """Allowed sentiment values."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(BaseModel):
    """Constrained sentiment output."""
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str] = Field(max_length=3)


class JobLevel(str, Enum):
    """Constrained job levels."""
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"


class JobClassification(BaseModel):
    """Constrained job classification."""
    level: JobLevel
    department: str = Field(pattern=r"^(Engineering|Data|Product|Design|Marketing|Sales|Operations)$")
    is_management: bool
    years_experience_min: int = Field(ge=0, le=30)


def analyze_sentiment_constrained(text: str) -> SentimentAnalysis:
    """Analyze sentiment with constrained output."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Analyze the sentiment of this text."},
            {"role": "user", "content": text}
        ],
        text_format=SentimentAnalysis
    )
    return response.output_parsed


def classify_job_constrained(title: str, description: str) -> JobClassification:
    """Classify job with constrained output."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Classify this job. Department must be one of: Engineering, Data, Product, Design, Marketing, Sales, Operations."
            },
            {"role": "user", "content": f"Title: {title}\n\nDescription: {description[:500]}"}
        ],
        text_format=JobClassification
    )
    return response.output_parsed


if __name__ == "__main__":
    print("=== CONSTRAINED GENERATION ===\n")
    
    # Sentiment analysis with constraints
    print("--- Sentiment Analysis ---")
    texts = [
        "Amazing opportunity with great benefits and flexible work!",
        "Must work weekends, no remote option, minimum wage.",
        "Standard developer position with typical requirements.",
    ]
    
    for text in texts:
        result = analyze_sentiment_constrained(text)
        print(f"\n\"{text[:50]}...\"")
        print(f"  Sentiment: {result.sentiment.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Key phrases: {result.key_phrases}")
    
    # Job classification with constraints
    print("\n\n--- Job Classification ---")
    jobs = [
        ("Senior Python Developer", "Build scalable backend systems. 5+ years experience."),
        ("Data Science Intern", "Learn ML techniques. No experience required."),
        ("VP of Engineering", "Lead 50+ engineers. 15 years experience."),
    ]
    
    for title, desc in jobs:
        result = classify_job_constrained(title, desc)
        print(f"\n{title}")
        print(f"  Level: {result.level.value}")
        print(f"  Department: {result.department}")
        print(f"  Management: {result.is_management}")
        print(f"  Min Experience: {result.years_experience_min} years")
    
    # Show that invalid values are prevented
    print("\n\n--- Constraint Benefits ---")
    print("• Sentiment must be: positive, negative, or neutral")
    print("• Confidence must be between 0.0 and 1.0")
    print("• Job level must be from predefined enum")
    print("• Department must match regex pattern")
