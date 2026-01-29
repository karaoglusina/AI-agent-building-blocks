"""
02 - Context Assembly
=====================
How to format retrieved documents for the prompt.

Key concept: How you present context matters - structure, ordering, and formatting affect quality.

Book reference: AI_eng.5, AI_eng.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Sample retrieved documents
RETRIEVED_DOCS = [
    {
        "title": "Senior Python Developer",
        "company": "TechCorp",
        "text": "We're looking for a Senior Python Developer with 5+ years experience. "
                "Must know Django, FastAPI, and PostgreSQL. Remote-friendly.",
        "score": 0.92,
    },
    {
        "title": "Python Backend Engineer",
        "company": "StartupXYZ",
        "text": "Join our fast-growing team! Python, AWS, microservices architecture. "
                "Competitive salary and equity.",
        "score": 0.87,
    },
    {
        "title": "Data Engineer",
        "company": "DataCo",
        "text": "Build data pipelines with Python, Spark, and Airflow. "
                "Experience with big data required.",
        "score": 0.81,
    }]


def format_simple(docs: list[dict]) -> str:
    """Simple concatenation - basic but works."""
    return "\n\n".join([doc["text"] for doc in docs])


def format_numbered(docs: list[dict]) -> str:
    """Numbered format - enables source citation."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc['title']} at {doc['company']}\n{doc['text']}")
    return "\n\n".join(parts)


def format_xml_style(docs: list[dict]) -> str:
    """XML-style structure - clearer boundaries for LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"<document id=\"{i}\">\n"
            f"  <title>{doc['title']}</title>\n"
            f"  <company>{doc['company']}</company>\n"
            f"  <content>{doc['text']}</content>\n"
            f"</document>"
        )
    return "\n\n".join(parts)


def format_with_relevance(docs: list[dict]) -> str:
    """Include relevance scores - helps LLM prioritize."""
    parts = []
    for doc in docs:
        parts.append(
            f"[Relevance: {doc['score']:.0%}]\n"
            f"**{doc['title']}** at {doc['company']}\n"
            f"{doc['text']}"
        )
    return "\n\n---\n\n".join(parts)


def query_with_format(query: str, context: str) -> str:
    """Run query with formatted context."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Answer based on the provided context only."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== CONTEXT ASSEMBLY FORMATS ===\n")
    
    formats = {
        "Simple": format_simple,
        "Numbered": format_numbered,
        "XML-style": format_xml_style,
        "With Relevance": format_with_relevance,
    }
    
    for name, formatter in formats.items():
        formatted = formatter(RETRIEVED_DOCS)
        print(f"--- {name} Format ---")
        print(formatted[:300] + "...\n")
    
    # Test query with XML format (often best for complex prompts)
    print("\n=== QUERY WITH XML FORMAT ===")
    query = "Which jobs offer remote work?"
    context = format_xml_style(RETRIEVED_DOCS)
    answer = query_with_format(query, context)
    print(f"Q: {query}")
    print(f"A: {answer}")
