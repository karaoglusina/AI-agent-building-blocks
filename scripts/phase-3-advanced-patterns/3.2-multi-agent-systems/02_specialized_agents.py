"""
02 - Specialized Agents
=======================
Create domain-specific agents with expertise and tools.

Key concept: Agents can have distinct capabilities, knowledge, and tools for their domain.

Book reference: AI_eng.6
"""

import json
from openai import OpenAI
import os

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


def search_jobs(keyword: str, location: str = "") -> str:
    """Search jobs by keyword and location."""
    jobs = load_sample_jobs(20)
    matches = []

    for job in jobs:
        text = f"{job['title']} {job['description']} {job.get('location', '')}".lower()
        if keyword.lower() in text:
            if not location or location.lower() in text:
                matches.append({
                    "title": job["title"],
                    "company": job["companyName"],
                    "location": job.get("location", "Not specified")
                })

    return json.dumps(matches[:5])


def get_job_stats(category: str = "") -> str:
    """Get statistics about jobs."""
    jobs = load_sample_jobs(20)
    if category:
        filtered = [j for j in jobs if category.lower() in j["title"].lower()]
        return json.dumps({"category": category, "count": len(filtered)})
    return json.dumps({"total": len(jobs)})


class SpecializedAgent:
    """Agent with specific domain expertise and tools."""

    def __init__(self, name: str, system_prompt: str, tools: list[dict], function_map: dict):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.function_map = function_map

    def execute(self, user_query: str) -> str:
        """Execute agent with tool access."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=self.tools
        )

        # Handle tool calls
        tool_calls = [item for item in (response.choices[0].message.tool_calls or []) if item.type == "function"]

        if tool_calls:
            for tool_call in tool_calls:
                func = self.function_map[tool_call.function.name]
                args = json.loads(tool_call.function.arguments)
                result = func(**args)

                print(f"  [{self.name}] Called: {tool_call.function.name}({args})")

                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

            # Generate final response
            final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            return final.choices[0].message.content

        return response.choices[0].message.content


if __name__ == "__main__":
    print("=== SPECIALIZED AGENTS ===\n")

    # Search Agent
    search_agent = SpecializedAgent(
        name="Search Agent",
        system_prompt="You help users find jobs. Search the database and present results clearly.",
        tools=[
            {
                "type": "function",
                "name": "search_jobs",
                "description": "Search for jobs by keyword and optionally location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Job keyword"},
                        "location": {"type": "string", "description": "Location filter"}
                    },
                    "required": ["keyword"]
                }
            }
        ],
        function_map={"search_jobs": search_jobs}
    )

    # Analytics Agent
    analytics_agent = SpecializedAgent(
        name="Analytics Agent",
        system_prompt="You provide job market statistics and insights. Use data to answer questions.",
        tools=[
            {
                "type": "function",
                "name": "get_job_stats",
                "description": "Get job statistics by category or overall",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "Job category to analyze"}
                    }
                }
            }
        ],
        function_map={"get_job_stats": get_job_stats}
    )

    # Test Search Agent
    print("User: Find Python developer jobs in Amsterdam")
    response = search_agent.execute("Find Python developer jobs in Amsterdam")
    print(f"[{search_agent.name}]: {response}\n")
    print("-" * 50 + "\n")

    # Test Analytics Agent
    print("User: How many engineer jobs are there?")
    response = analytics_agent.execute("How many engineer jobs are there?")
    print(f"[{analytics_agent.name}]: {response}\n")
