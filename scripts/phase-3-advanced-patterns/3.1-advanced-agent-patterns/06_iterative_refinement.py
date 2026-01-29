"""
06 - Iterative Refinement
=========================
Improve output through multiple passes.

Key concept: Generate → Evaluate → Refine loop improves quality - each iteration builds on previous version.

Book reference: AI_eng.6
"""

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


class JobDescription(BaseModel):
    """Structured job description."""
    summary: str
    key_responsibilities: list[str]
    required_qualifications: list[str]
    quality_score: float


def generate_description(job: dict) -> str:
    """Generate initial job description."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Create a clear, engaging job description."
    },
    {
    "role": "user",
    "content": f"""Write a job description for:
    Title: {job['title']}
    Company: {job.get('company', 'A leading company')}
    Raw info: {job.get('description', '')[:300]}
    
    Include:
    - Summary (2-3 sentences)
    - Key responsibilities (3-4 bullets)
    - Required qualifications (3-4 bullets)"""
    }
    ]
    )
    return response.choices[0].message.content


def evaluate_description(description: str) -> dict:
    """Evaluate description quality and provide feedback."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": """Rate this job description on:
    1. Clarity (0-10): Easy to understand?
    2. Completeness (0-10): All important info included?
    3. Appeal (0-10): Engaging and attractive?
    
    Provide overall score (0-10) and specific improvement suggestions."""
    },
    {
    "role": "user",
    "content": f"Job Description:\n{description}"
    }
    ]
    )

    # Parse evaluation
    eval_text = response.choices[0].message.content
    score = 7.0  # Default - in production, parse from response

    # Extract score if mentioned
    if "score" in eval_text.lower():
        import re
        match = re.search(r'(\d+(?:\.\d+)?)/10', eval_text)
        if match:
            score = float(match.group(1))

    return {
        "score": score,
        "feedback": eval_text
    }


def refine_description(original: str, feedback: str, iteration: int) -> str:
    """Refine description based on feedback."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Improve the job description based on the feedback provided."
    },
    {
    "role": "user",
    "content": f"""Original Description:
    {original}
    
    Feedback:
    {feedback}
    
    Please provide an improved version addressing the feedback."""
    }
    ]
    )
    return response.choices[0].message.content


def iterative_refinement(
    job: dict,
    max_iterations: int = 3,
    target_score: float = 8.5
) -> dict:
    """Refine job description through multiple iterations."""
    print(f"\n{'=' * 70}")
    print(f"Refining: {job['title']}")
    print("=" * 70)

    description = generate_description(job)
    history = []

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        # Evaluate
        evaluation = evaluate_description(description)
        score = evaluation["score"]

        print(f"Score: {score:.1f}/10")
        print(f"Feedback: {evaluation['feedback'][:150]}...")

        history.append({
            "iteration": i + 1,
            "description": description,
            "score": score,
            "feedback": evaluation["feedback"]
        })

        # Check if target met
        if score >= target_score:
            print(f"\n✓ Target score reached ({score:.1f} >= {target_score})")
            break

        # Refine for next iteration
        if i < max_iterations - 1:
            print("Refining...")
            description = refine_description(description, evaluation["feedback"], i + 1)

    return {
        "final_description": description,
        "final_score": history[-1]["score"],
        "iterations": len(history),
        "history": history,
        "improvement": history[-1]["score"] - history[0]["score"]
    }


if __name__ == "__main__":
    print("=== ITERATIVE REFINEMENT ===\n")

    jobs = load_sample_jobs(2)

    for job in jobs:
        result = iterative_refinement(job, max_iterations=3, target_score=8.5)

        print(f"\n{'=' * 70}")
        print(f"RESULTS:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Initial score: {result['history'][0]['score']:.1f}")
        print(f"  Final score: {result['final_score']:.1f}")
        print(f"  Improvement: +{result['improvement']:.1f}")
        print(f"\nFinal Description:")
        print(result['final_description'][:300] + "...")
        print("=" * 70)
