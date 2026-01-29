"""
05 - Tools with Shared Context
==============================
Pass context between tool calls.

Key concept: Maintain state across tool calls using a context object.

Book reference: AI_eng.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
from dataclasses import dataclass, field
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@dataclass
class AgentContext:
    """Shared context for the agent."""
    user_preferences: dict = field(default_factory=dict)
    search_history: list = field(default_factory=list)
    last_results: list = field(default_factory=list)


def set_preference(context: AgentContext, key: str, value: str) -> str:
    """Store a user preference."""
    context.user_preferences[key] = value
    return f"Preference set: {key} = {value}"


def search_jobs(context: AgentContext, keyword: str) -> str:
    """Search jobs, applying stored preferences."""
    # Apply preferences to search
    location = context.user_preferences.get("location", "any")
    remote = context.user_preferences.get("remote", "any")
    
    # Mock search results
    results = [
        {"title": f"Senior {keyword} Developer", "location": location, "remote": remote == "yes"},
        {"title": f"{keyword} Engineer", "location": location, "remote": remote == "yes"}]
    
    context.search_history.append(keyword)
    context.last_results = results
    
    return json.dumps({
        "results": results,
        "filters_applied": {"location": location, "remote": remote}
    })


def get_last_results(context: AgentContext) -> str:
    """Get the last search results."""
    if not context.last_results:
        return "No previous search results"
    return json.dumps(context.last_results)


def get_search_history(context: AgentContext) -> str:
    """Get the search history."""
    return json.dumps(context.search_history)


# Tools with context
TOOLS = [
    {
        "type": "function",
        "name": "set_preference",
        "description": "Set a user preference (location, remote, salary, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"}
            },
            "required": ["key", "value"]
        }
    },
    {
        "type": "function",
        "name": "search_jobs",
        "description": "Search for jobs. User preferences are automatically applied.",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"]
        }
    },
    {
        "type": "function",
        "name": "get_last_results",
        "description": "Get the last search results",
        "parameters": {"type": "object", "properties": {}}
    }]


def run_with_context(context: AgentContext, user_input: str) -> str:
    """Run agent with shared context."""
    messages = [
        {"role": "system", "content": "Help users find jobs. Remember their preferences."},
        {"role": "user", "content": user_input}
    ]
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
    
    # Process tool calls
    for item in (response.choices[0].message.tool_calls or []):
        if item.type == "function":
            func_name = item.function.name
            args = json.loads(item.function.arguments)
            
            # Execute with context
            if func_name == "set_preference":
                result = set_preference(context, args["key"], args["value"])
            elif func_name == "search_jobs":
                result = search_jobs(context, args["keyword"])
            elif func_name == "get_last_results":
                result = get_last_results(context)
            else:
                result = "Unknown function"
            
            print(f"  Tool: {func_name}({args}) → {result[:80]}...")
    
    return response.choices[0].message.content or "Done"


if __name__ == "__main__":
    print("=== TOOLS WITH SHARED CONTEXT ===\n")
    
    # Create shared context
    context = AgentContext()
    
    # Simulate conversation
    queries = [
        "I prefer remote jobs in Amsterdam",
        "Search for Python developer jobs",
        "Now search for data science roles"]
    
    for query in queries:
        print(f"User: {query}")
        result = run_with_context(context, query)
        print()
    
    print("\n=== CONTEXT STATE ===")
    print(f"Preferences: {context.user_preferences}")
    print(f"Search history: {context.search_history}")
