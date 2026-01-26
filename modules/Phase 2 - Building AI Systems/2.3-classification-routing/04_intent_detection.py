"""
04 - Intent Detection
=====================
Detect user intent from natural language queries.

Key concept: Intent detection routes user requests to the right handler.

Book reference: speach_lang.II.15.3
"""

from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

client = OpenAI()


class Intent(str, Enum):
    """Possible user intents for a job search system."""
    SEARCH = "search"           # Looking for jobs
    COMPARE = "compare"         # Compare jobs or companies
    DETAILS = "details"         # Get job details
    APPLY = "apply"             # Apply to a job
    SALARY = "salary"           # Salary information
    COMPANY = "company"         # Company information
    SKILLS = "skills"           # Skills requirements
    CLARIFY = "clarify"         # Unclear, need more info


class DetectedIntent(BaseModel):
    """Result of intent detection."""
    intent: Intent
    entities: list[str]  # Extracted entities (job titles, companies, locations)
    confidence: float


def detect_intent(query: str) -> DetectedIntent:
    """Detect user intent from a query."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": """Detect the user's intent from their query about jobs.
Intents: search, compare, details, apply, salary, company, skills, clarify.
Extract any mentioned job titles, companies, locations, or skills."""
            },
            {"role": "user", "content": query}
        ],
        text_format=DetectedIntent
    )
    return response.output_parsed


# Sample queries to test
TEST_QUERIES = [
    "Find me Python developer jobs in Amsterdam",
    "What's the salary range for data scientists at Google?",
    "Compare Amazon and Microsoft for engineering roles",
    "Tell me more about that senior position at Spotify",
    "How do I apply for the job at Netflix?",
    "What skills do I need for a machine learning role?",
    "hmm not sure",
]


if __name__ == "__main__":
    print("=== INTENT DETECTION ===\n")
    
    for query in TEST_QUERIES:
        result = detect_intent(query)
        print(f"Query: \"{query}\"")
        print(f"  Intent: {result.intent.value}")
        print(f"  Entities: {result.entities}")
        print(f"  Confidence: {result.confidence:.0%}")
        print()
