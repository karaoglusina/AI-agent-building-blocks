"""
01 - ReAct Pattern
==================
Reasoning + Acting interleaved for better decision making.

Key concept: Agents alternate between reasoning (think) and acting (tool use) - explicit reasoning improves tool selection and error recovery.

Book reference: hands_on_LLM.II.7, NLP_cook.10
"""

import json
from openai import OpenAI
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()

# Define tools
TOOLS = [
    {
        "type": "function",
        "name": "search_jobs",
        "description": "Search jobs by keyword in title or description",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Keyword to search for"}
            },
            "required": ["keyword"]
        }
    },
    {
        "type": "function",
        "name": "filter_by_location",
        "description": "Filter jobs by location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Location to filter by"}
            },
            "required": ["location"]
        }
    }
]


def search_jobs(keyword: str) -> str:
    """Search jobs by keyword."""
    jobs = load_sample_jobs(50)
    keyword_lower = keyword.lower()
    matches = [
        {"title": job["title"], "company": job.get("company", "Unknown")}
        for job in jobs
        if keyword_lower in job["title"].lower() or keyword_lower in job.get("description", "").lower()
    ]
    return json.dumps(matches[:5])


def filter_by_location(location: str) -> str:
    """Filter jobs by location."""
    jobs = load_sample_jobs(50)
    matches = [
        {"title": job["title"], "location": job.get("location", "Remote")}
        for job in jobs
        if location.lower() in job.get("location", "").lower()
    ]
    return json.dumps(matches[:5])


FUNCTION_MAP = {
    "search_jobs": search_jobs,
    "filter_by_location": filter_by_location,
}


def run_react_agent(query: str, max_iterations: int = 5) -> str:
    """Run agent with ReAct pattern: Reason → Act → Observe loop."""
    messages = [
        {
            "role": "system",
            "content": """You are a job search assistant using the ReAct pattern.
Before each action, think through:
1. What information do I need?
2. Which tool should I use?
3. What parameters make sense?

Format your reasoning explicitly in your response."""
        },
        {"role": "user", "content": query}
    ]

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        response = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            tools=TOOLS
        )

        # Check for tool calls
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            print("THOUGHT: Task complete, providing final answer")
            return response.output_text

        # Execute tool calls
        for tool_call in tool_calls:
            func_name = tool_call.name
            func_args = json.loads(tool_call.arguments)

            print(f"ACTION: {func_name}({func_args})")
            result = FUNCTION_MAP[func_name](**func_args)
            print(f"OBSERVATION: {result[:200]}...")

            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": result
            })

    return "Max iterations reached"


if __name__ == "__main__":
    print("=== REACT PATTERN ===\n")

    queries = [
        "Find Python developer jobs in San Francisco",
        "What engineering jobs are available remotely?"
    ]

    for query in queries:
        print(f"\nQUERY: {query}")
        print("=" * 70)
        result = run_react_agent(query)
        print(f"\nFINAL ANSWER: {result}")
        print("\n" + "=" * 70)
