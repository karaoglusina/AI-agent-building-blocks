"""
03 - Refinement Chain
=====================
Iteratively improve output through multiple LLM passes.

Key concept: Each pass refines the previous output based on feedback - converges to higher quality than single-shot generation.

Book reference: AI_eng.6
"""

from openai import OpenAI
import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


def refine_output(content: str, feedback: str) -> str:
    """Apply refinement based on feedback."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You improve content based on feedback. Keep what works, fix what doesn't."
            },
            {
                "role": "user",
                "content": f"""Original content:
{content}

Feedback:
{feedback}

Please provide an improved version that addresses the feedback."""
            }
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def generate_feedback(content: str, criteria: list[str]) -> str:
    """Generate constructive feedback based on criteria."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a constructive critic. Provide specific, actionable feedback."
            },
            {
                "role": "user",
                "content": f"""Evaluate this content against these criteria:
{chr(10).join(f'- {c}' for c in criteria)}

Content:
{content}

Provide specific feedback on what to improve."""
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


def refinement_chain(initial_content: str, criteria: list[str], iterations: int = 3) -> list[dict]:
    """Run refinement chain: Generate → Critique → Refine loop."""
    history = [{"version": 0, "content": initial_content, "feedback": "Initial version"}]

    current_content = initial_content

    for i in range(iterations):
        print(f"\n=== ITERATION {i + 1} ===")

        # Get feedback
        feedback = generate_feedback(current_content, criteria)
        print(f"Feedback: {feedback[:200]}...")

        # Refine based on feedback
        refined = refine_output(current_content, feedback)
        print(f"Refined: {refined[:200]}...")

        history.append({
            "version": i + 1,
            "content": refined,
            "feedback": feedback
        })

        current_content = refined

    return history


if __name__ == "__main__":
    # Load a job and create initial description
    jobs = load_sample_jobs(1)
    job = jobs[0]

    # Generate initial job summary
    initial_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Write a compelling 2-paragraph summary for this job:\n\nTitle: {job['title']}\nCompany: {job.get('company', 'Unknown')}\nDescription: {job.get('description', '')[:500]}"
            }
        ],
        temperature=0.7
    )
    initial_summary = initial_response.choices[0].message.content

    print("=== INITIAL SUMMARY ===")
    print(initial_summary)
    print("\n" + "=" * 70)

    # Define quality criteria
    criteria = [
        "Clear and concise language",
        "Highlights key responsibilities",
        "Emphasizes unique selling points",
        "Professional tone",
        "Engaging and compelling"
    ]

    # Run refinement chain
    history = refinement_chain(initial_summary, criteria, iterations=2)

    print("\n" + "=" * 70)
    print("=== FINAL VERSION ===")
    print(history[-1]["content"])

    print(f"\n\nTotal versions: {len(history)}")
    print("Quality improved through iterative refinement!")
