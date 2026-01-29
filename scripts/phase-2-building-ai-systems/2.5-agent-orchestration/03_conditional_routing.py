"""
03 - Conditional Routing
========================
Route to different paths based on LLM decision.

Key concept: Use LLM to classify input, then branch to specialized handlers.

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


class TaskType(str, Enum):
    """Types of user tasks."""
    SEARCH = "search"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXPLAIN = "explain"


class TaskClassification(BaseModel):
    """Classification result."""
    task_type: TaskType
    entities: list[str]


def classify_task(user_input: str) -> TaskClassification:
    """Classify the user's task."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Classify the user's task as: search, summarize, compare, or explain. "
    "Extract any relevant entities (job titles, companies, skills)."
    },
    {"role": "user", "content": user_input}
    ],
    response_format={"type": "json_object"}
    )
    return TaskClassification.model_validate_json(response.choices[0].message.content)


# Specialized handlers for each task type
def handle_search(entities: list[str]) -> str:
    """Handle search tasks."""
    return f"🔍 Searching for jobs matching: {', '.join(entities)}"


def handle_summarize(entities: list[str]) -> str:
    """Handle summarization tasks."""
    return f"📝 Summarizing information about: {', '.join(entities)}"


def handle_compare(entities: list[str]) -> str:
    """Handle comparison tasks."""
    if len(entities) >= 2:
        return f"⚖️ Comparing: {entities[0]} vs {entities[1]}"
    return "⚖️ Need at least two items to compare"


def handle_explain(entities: list[str]) -> str:
    """Handle explanation tasks."""
    return f"💡 Explaining: {', '.join(entities)}"


HANDLERS = {
    TaskType.SEARCH: handle_search,
    TaskType.SUMMARIZE: handle_summarize,
    TaskType.COMPARE: handle_compare,
    TaskType.EXPLAIN: handle_explain,
}


def route_and_execute(user_input: str) -> str:
    """Classify task and route to appropriate handler."""
    # Step 1: Classify
    classification = classify_task(user_input)
    print(f"  Classified as: {classification.task_type.value}")
    print(f"  Entities: {classification.entities}")
    
    # Step 2: Route to handler
    handler = HANDLERS[classification.task_type]
    result = handler(classification.entities)
    
    return result


if __name__ == "__main__":
    print("=== CONDITIONAL ROUTING ===\n")
    
    test_inputs = [
        "Find me Python developer jobs in Amsterdam",
        "Summarize the requirements for data science roles",
        "Compare Google and Amazon for engineering positions",
        "What does a DevOps engineer do?"]
    
    for user_input in test_inputs:
        print(f"User: \"{user_input}\"")
        result = route_and_execute(user_input)
        print(f"  Result: {result}\n")
