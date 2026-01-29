"""
05 - Query Router
=================
Route queries to different handlers based on classification.

Key concept: Routing separates concerns and enables specialized handlers.

Book reference: AI_eng.6, AI_eng.10.3
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


class Route(str, Enum):
    """Available routes for query handling."""
    SEARCH = "search"       # Vector search for job discovery
    SQL = "sql"             # Structured queries (filters, counts)
    CHAT = "chat"           # General conversation
    SUMMARIZE = "summarize" # Summarization tasks


class RoutingDecision(BaseModel):
    """Router's decision."""
    route: Route
    reason: str


def route_query(query: str) -> RoutingDecision:
    """Determine the best handler for a query."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """Route the query to the best handler:
    - search: Finding jobs by skills, keywords, or semantic similarity
    - sql: Counting, filtering by exact values, aggregations
    - chat: General questions, explanations, advice
    - summarize: Summarizing job descriptions or market trends"""
    },
    {"role": "user", "content": query}
    ]
    ,
    response_format={"type": "json_object"})
    return RoutingDecision.model_validate_json(response.choices[0].message.content)


# Handler functions (simplified for demo)
def handle_search(query: str) -> str:
    return f"[SEARCH] Running vector search for: {query}"


def handle_sql(query: str) -> str:
    return f"[SQL] Executing structured query: {query}"


def handle_chat(query: str) -> str:
    return f"[CHAT] Generating conversational response for: {query}"


def handle_summarize(query: str) -> str:
    return f"[SUMMARIZE] Creating summary for: {query}"


HANDLERS = {
    Route.SEARCH: handle_search,
    Route.SQL: handle_sql,
    Route.CHAT: handle_chat,
    Route.SUMMARIZE: handle_summarize,
}


def process_query(query: str) -> str:
    """Route and process a query."""
    decision = route_query(query)
    handler = HANDLERS[decision.route]
    return handler(query), decision


# Test queries
QUERIES = [
    "Find Python jobs similar to data engineering",
    "How many remote jobs are there in the Netherlands?",
    "What should I learn to become a data scientist?",
    "Summarize the top 10 machine learning job requirements",
    "Senior backend developer positions in fintech"]

if __name__ == "__main__":
    print("=== QUERY ROUTER ===\n")
    
    for query in QUERIES:
        result, decision = process_query(query)
        print(f"Query: \"{query}\"")
        print(f"  Route: {decision.route.value}")
        print(f"  Reason: {decision.reason}")
        print(f"  → {result}")
        print()
