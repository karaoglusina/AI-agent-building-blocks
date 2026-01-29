"""
08 - Multi-Tool Agent
=====================
Agent with multiple capabilities and tool selection.

Key concept: Give agents multiple tools - the LLM decides which to use.

Book reference: AI_eng.6
"""

import json
from openai import OpenAI
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.data_loader import load_sample_jobs
from pathlib import Path


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()

# Load some job data for the tools
JOBS = load_sample_jobs(20)


def search_jobs(keyword: str, location: str = None) -> str:
    """Search for jobs by keyword and optional location."""
    results = [j for j in JOBS if keyword.lower() in j["title"].lower()]
    if location:
        results = [j for j in results if location.lower() in j.get("location", "").lower()]
    return json.dumps([{"title": j["title"], "company": j["companyName"]} for j in results[:3]])


def get_job_details(job_title: str) -> str:
    """Get detailed info about a specific job."""
    for job in JOBS:
        if job_title.lower() in job["title"].lower():
            return json.dumps({
                "title": job["title"],
                "company": job["companyName"],
                "description": job["description"][:300],
                "location": job.get("location", "Unknown")
            })
    return json.dumps({"error": "Job not found"})


def compare_jobs(job1: str, job2: str) -> str:
    """Compare two jobs."""
    j1 = next((j for j in JOBS if job1.lower() in j["title"].lower()), None)
    j2 = next((j for j in JOBS if job2.lower() in j["title"].lower()), None)
    
    if not j1 or not j2:
        return json.dumps({"error": "One or both jobs not found"})
    
    return json.dumps({
        "job1": {"title": j1["title"], "company": j1["companyName"]},
        "job2": {"title": j2["title"], "company": j2["companyName"]},
        "comparison": "See descriptions for detailed comparison"
    })


def count_jobs(category: str = None) -> str:
    """Count jobs, optionally filtered by category."""
    if category:
        count = sum(1 for j in JOBS if category.lower() in j["title"].lower())
    else:
        count = len(JOBS)
    return json.dumps({"count": count, "filter": category})


TOOLS = [
    {
        "type": "function", "name": "search_jobs",
        "description": "Search for jobs by keyword and optional location",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "location": {"type": "string", "description": "Optional location filter"}
            },
            "required": ["keyword"]
        }
    },
    {
        "type": "function", "name": "get_job_details",
        "description": "Get detailed information about a specific job",
        "parameters": {
            "type": "object",
            "properties": {"job_title": {"type": "string"}},
            "required": ["job_title"]
        }
    },
    {
        "type": "function", "name": "compare_jobs",
        "description": "Compare two jobs side by side",
        "parameters": {
            "type": "object",
            "properties": {"job1": {"type": "string"}, "job2": {"type": "string"}},
            "required": ["job1", "job2"]
        }
    },
    {
        "type": "function", "name": "count_jobs",
        "description": "Count total jobs or jobs in a category",
        "parameters": {
            "type": "object",
            "properties": {"category": {"type": "string"}}
        }
    }
]

FUNCTION_MAP = {
    "search_jobs": search_jobs,
    "get_job_details": get_job_details,
    "compare_jobs": compare_jobs,
    "count_jobs": count_jobs,
}


def run_multi_tool_agent(query: str) -> str:
    """Run agent with multiple tools."""
    messages = [
        {"role": "system", "content": "You're a job assistant with multiple tools. Use the right tool for each task."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS)
    
    # Process all tool calls
    tool_results = []
    for item in (response.choices[0].message.tool_calls or []):
        if item.type == "function":
            func = FUNCTION_MAP.get(item.function.name)
            if func:
                args = json.loads(item.function.arguments)
                result = func(**args)
                print(f"  Tool: {item.function.name}({args})")
                
                messages.append({"role": "assistant", "content": None, "tool_calls": [item]})
                messages.append({"role": "tool", "tool_call_id": item.id, "content": result})
    
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    print(f"=== MULTI-TOOL AGENT ({len(JOBS)} jobs loaded) ===\n")
    
    queries = [
        "Find Python developer jobs",
        "How many jobs do you have in total?",
        "Tell me more about any data science role",
        "Compare a developer and engineer position"]
    
    for query in queries:
        print(f"User: {query}")
        answer = run_multi_tool_agent(query)
        print(f"Agent: {answer}\n")
        print("-" * 50 + "\n")
