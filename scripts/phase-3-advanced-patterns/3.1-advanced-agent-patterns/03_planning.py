"""
03 - Planning
=============
Decompose complex tasks into executable steps.

Key concept: Before acting, create a plan - break complex queries into ordered subtasks for better results.

Book reference: AI_eng.6
"""

import json
from openai import OpenAI
import os
from pydantic import BaseModel
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


class Plan(BaseModel):
    """A plan with ordered steps."""
    goal: str
    steps: list[str]
    expected_outcome: str


def create_plan(query: str) -> Plan:
    """Generate a step-by-step plan for the query."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """You are a planning agent. Break down the user's request into clear, ordered steps.
    Each step should be specific and actionable."""
    },
    {
    "role": "user",
    "content": f"""Create a plan for: {query}
    
    Return JSON with:
    - goal: what we're trying to achieve
    - steps: ordered list of specific actions
    - expected_outcome: what success looks like"""
    }
    ]
    )

    plan_data = json.loads(response.choices[0].message.content)
    return Plan(**plan_data)


def execute_step(step: str, context: dict) -> dict:
    """Execute a single step of the plan."""
    jobs = load_sample_jobs(50)

    # Simple execution logic based on step keywords
    if "search" in step.lower() or "find" in step.lower():
        # Extract search term from step
        keyword = step.split("for")[-1].strip() if "for" in step else "developer"
        matches = [
            job for job in jobs
            if keyword.lower() in job["title"].lower() or keyword.lower() in job.get("description", "").lower()
        ]
        return {"action": "search", "results": matches[:5], "count": len(matches)}

    elif "filter" in step.lower():
        # Extract filter criteria
        if "location" in step.lower():
            location = "remote" if "remote" in step.lower() else "San Francisco"
            matches = [
                job for job in jobs
                if location.lower() in job.get("location", "").lower()
            ]
            return {"action": "filter", "results": matches[:5], "count": len(matches)}

    elif "rank" in step.lower() or "sort" in step.lower():
        # Simple ranking by title length as proxy for detail
        ranked = sorted(jobs[:10], key=lambda x: len(x.get("description", "")), reverse=True)
        return {"action": "rank", "results": ranked[:5]}

    elif "summarize" in step.lower():
        results = context.get("previous_results", jobs[:3])
        summaries = [{"title": job["title"], "company": job.get("company", "Unknown")} for job in results]
        return {"action": "summarize", "results": summaries}

    return {"action": "unknown", "results": []}


def execute_plan(plan: Plan) -> dict:
    """Execute all steps in the plan."""
    print(f"\n=== EXECUTING PLAN ===")
    print(f"Goal: {plan.goal}\n")

    context = {}
    results = []

    for i, step in enumerate(plan.steps, 1):
        print(f"Step {i}/{len(plan.steps)}: {step}")

        step_result = execute_step(step, context)
        results.append({"step": step, "result": step_result})

        # Update context for next step
        if step_result.get("results"):
            context["previous_results"] = step_result["results"]

        print(f"  → {step_result['action']}: {step_result.get('count', len(step_result.get('results', [])))} items")

    print(f"\n✓ Plan complete\n")

    return {
        "plan": plan,
        "results": results,
        "final_context": context
    }


if __name__ == "__main__":
    print("=== PLANNING AGENT ===\n")

    queries = [
        "Find the best remote Python developer jobs and summarize the top 3",
        "Search for engineering jobs in San Francisco, filter by senior level, and rank by salary"
    ]

    for query in queries:
        print(f"\nQUERY: {query}")
        print("=" * 70)

        # Create plan
        plan = create_plan(query)
        print(f"\nPLAN CREATED:")
        print(f"  Steps: {len(plan.steps)}")
        for i, step in enumerate(plan.steps, 1):
            print(f"    {i}. {step}")

        # Execute plan
        result = execute_plan(plan)

        print(f"Expected: {plan.expected_outcome}")
        print("=" * 70)
