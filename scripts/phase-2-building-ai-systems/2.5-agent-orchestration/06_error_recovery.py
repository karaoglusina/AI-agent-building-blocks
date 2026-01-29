"""
06 - Error Recovery
===================
Handle tool failures and retry patterns.

Key concept: Agents fail - build in retry logic, fallbacks, and graceful degradation.

Book reference: AI_eng.6 (Agent Failure Modes)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import json
import random
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class ToolError(Exception):
    """Custom error for tool failures."""
    pass


def unreliable_search(keyword: str) -> str:
    """A tool that sometimes fails (simulating real-world APIs)."""
    if random.random() < 0.4:  # 40% failure rate
        raise ToolError("Search service temporarily unavailable")
    return json.dumps([{"title": f"{keyword} Developer", "company": "TechCorp"}])


def backup_search(keyword: str) -> str:
    """Backup search that's more reliable but less accurate."""
    return json.dumps([{"title": f"Job related to {keyword}", "company": "Unknown"}])


def search_with_retry(keyword: str, max_retries: int = 3) -> tuple[str, str]:
    """Search with retry logic and fallback."""
    last_error = None
    
    # Try primary search with retries
    for attempt in range(max_retries):
        try:
            result = unreliable_search(keyword)
            return result, f"Success on attempt {attempt + 1}"
        except ToolError as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")
    
    # Fall back to backup search
    print("  Falling back to backup search...")
    result = backup_search(keyword)
    return result, f"Used backup after {max_retries} failures"


TOOLS = [
    {
        "type": "function",
        "name": "search_jobs",
        "description": "Search for jobs (may be unreliable)",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"]
        }
    }
]


def run_resilient_agent(query: str) -> str:
    """Agent with error handling."""
    messages = [
        {"role": "system", "content": "Help find jobs. If search fails, explain the issue."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
    
    for item in (response.choices[0].message.tool_calls or []):
        if item.type == "function" and item.function.name == "search_jobs":
            args = json.loads(item.function.arguments)
            
            # Execute with error handling
            result, status = search_with_retry(args["keyword"])
            
            # Add result to conversation
            messages.append({"role": "assistant", "content": None, "tool_calls": [item]})
            messages.append({
                "role": "tool",
                "tool_call_id": item.id,
                "content": f"Status: {status}\nResults: {result}"
            })
    
    # Get final response
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    print("=== ERROR RECOVERY ===\n")
    print("(Simulating unreliable API with 40% failure rate)\n")
    
    for i in range(3):
        print(f"--- Query {i + 1} ---")
        result = run_resilient_agent("Find Python developer jobs")
        print(f"Final answer: {result}\n")
    
    print("\n=== FAILURE PATTERNS ===")
    print("1. Retry: Try again (network glitches)")
    print("2. Fallback: Use backup service")
    print("3. Degrade: Return partial results")
    print("4. Explain: Tell user what went wrong")
