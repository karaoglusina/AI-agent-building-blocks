"""
03 - Conditional Routing
========================
Route to different paths based on LLM decision.

Key concept: Use LLM to classify input, then branch to specialized handlers.

Book reference: AI_eng.6, AI_eng.10.3
"""

from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

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
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Classify the user's task as: search, summarize, compare, or explain. "
                           "Extract any relevant entities (job titles, companies, skills)."
            },
            {"role": "user", "content": user_input}
        ],
        text_format=TaskClassification
    )
    return response.output_parsed


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
        "What does a DevOps engineer do?",
    ]
    
    for user_input in test_inputs:
        print(f"User: \"{user_input}\"")
        result = route_and_execute(user_input)
        print(f"  Result: {result}\n")
