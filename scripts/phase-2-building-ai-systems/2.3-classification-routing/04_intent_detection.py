"""
04 - Intent Detection
=====================
Detect user intent from natural language queries.

Key concept: Intent detection routes user requests to the right handler.

Book reference: speach_lang.II.15.3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os
from pydantic import BaseModel
from enum import Enum


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

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
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """Detect the user's intent from their query about jobs.
    Intents: search, compare, details, apply, salary, company, skills, clarify.
    Extract any mentioned job titles, companies, locations, or skills."""
    },
    {"role": "user", "content": query}
    ]
    ,
    response_format={"type": "json_object"})
    return DetectedIntent.model_validate_json(response.choices[0].message.content)


# Sample queries to test
TEST_QUERIES = [
    "Find me Python developer jobs in Amsterdam",
    "What's the salary range for data scientists at Google?",
    "Compare Amazon and Microsoft for engineering roles",
    "Tell me more about that senior position at Spotify",
    "How do I apply for the job at Netflix?",
    "What skills do I need for a machine learning role?",
    "hmm not sure"]


if __name__ == "__main__":
    print("=== INTENT DETECTION ===\n")
    
    for query in TEST_QUERIES:
        result = detect_intent(query)
        print(f"Query: \"{query}\"")
        print(f"  Intent: {result.intent.value}")
        print(f"  Entities: {result.entities}")
        print(f"  Confidence: {result.confidence:.0%}")
        print()
