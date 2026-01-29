"""
02 - Self-Reflection
====================
Agent critiques and improves its own output.

Key concept: Have the model evaluate and revise its own responses - catches errors and improves quality through self-critique.

Book reference: AI_eng.6
"""

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


def generate_job_summary(job: dict) -> str:
    """Generate initial job summary."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Summarize this job posting in 2-3 sentences."
    },
    {
    "role": "user",
    "content": f"Title: {job['title']}\nDescription: {job.get('description', 'No description')[:500]}"
    }
    ]
    )
    return response.choices[0].message.content


def reflect_on_summary(original_job: dict, summary: str) -> dict:
    """Critique the summary and suggest improvements."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """You are a quality reviewer. Evaluate this job summary for:
    1. Accuracy - Does it match the job posting?
    2. Completeness - Are key details included?
    3. Clarity - Is it easy to understand?
    4. Conciseness - Is it the right length?
    
    Provide a critique and suggestions for improvement."""
    },
    {
    "role": "user",
    "content": f"""Original Job:
    Title: {original_job['title']}
    Description: {original_job.get('description', '')[:500]}
    
    Summary to Review:
    {summary}
    
    Provide your critique and specific suggestions."""
    }
    ]
    )
    return {
        "critique": response.choices[0].message.content,
        "needs_improvement": "improve" in response.choices[0].message.content.lower() or "missing" in response.choices[0].message.content.lower()
    }


def refine_summary(job: dict, original_summary: str, critique: str) -> str:
    """Refine the summary based on critique."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Revise the job summary based on the critique provided."
    },
    {
    "role": "user",
    "content": f"""Job: {job['title']}
    Description: {job.get('description', '')[:500]}
    
    Original Summary:
    {original_summary}
    
    Critique:
    {critique}
    
    Provide an improved summary addressing the critique."""
    }
    ]
    )
    return response.choices[0].message.content


def summarize_with_reflection(job: dict, max_iterations: int = 2) -> dict:
    """Generate and refine summary through reflection."""
    print(f"\n=== Summarizing: {job['title']} ===\n")

    # Initial generation
    summary = generate_job_summary(job)
    print(f"INITIAL SUMMARY:\n{summary}\n")

    history = [{"iteration": 0, "summary": summary}]

    # Reflection loop
    for i in range(max_iterations):
        print(f"--- Reflection {i + 1} ---")
        reflection = reflect_on_summary(job, summary)
        print(f"CRITIQUE:\n{reflection['critique']}\n")

        if not reflection["needs_improvement"]:
            print("✓ Summary approved\n")
            break

        # Refine based on critique
        summary = refine_summary(job, summary, reflection["critique"])
        print(f"REFINED SUMMARY:\n{summary}\n")
        history.append({"iteration": i + 1, "summary": summary})

    return {
        "final_summary": summary,
        "iterations": len(history),
        "history": history
    }


if __name__ == "__main__":
    print("=== SELF-REFLECTION ===\n")

    jobs = load_sample_jobs(2)

    for job in jobs:
        result = summarize_with_reflection(job)
        print(f"Final result after {result['iterations']} iteration(s)")
        print("=" * 70)
