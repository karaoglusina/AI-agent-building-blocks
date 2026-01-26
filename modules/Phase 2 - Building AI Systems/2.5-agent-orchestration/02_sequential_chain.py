"""
02 - Sequential Chain
=====================
Chain multiple LLM calls in sequence.

Key concept: Break complex tasks into steps, passing output from one to the next.

Book reference: hands_on_LLM.II.7
"""

from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


def extract_skills(job_description: str) -> str:
    """Step 1: Extract skills from job description."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Extract a comma-separated list of technical skills."},
            {"role": "user", "content": job_description}
        ]
    )
    return response.output_text


def categorize_skills(skills: str) -> str:
    """Step 2: Categorize the extracted skills."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Categorize these skills into: Programming, Databases, Cloud, Tools, Soft Skills"
            },
            {"role": "user", "content": skills}
        ]
    )
    return response.output_text


def generate_learning_path(categorized_skills: str) -> str:
    """Step 3: Generate a learning path."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Create a brief learning path (3-5 steps) to acquire these skills."
            },
            {"role": "user", "content": categorized_skills}
        ]
    )
    return response.output_text


def sequential_chain(job_description: str) -> dict:
    """Run the full sequential chain."""
    print("Step 1: Extracting skills...")
    skills = extract_skills(job_description)
    print(f"  → {skills[:100]}...")
    
    print("\nStep 2: Categorizing skills...")
    categorized = categorize_skills(skills)
    print(f"  → {categorized[:100]}...")
    
    print("\nStep 3: Generating learning path...")
    learning_path = generate_learning_path(categorized)
    
    return {
        "skills": skills,
        "categorized": categorized,
        "learning_path": learning_path
    }


if __name__ == "__main__":
    print("=== SEQUENTIAL CHAIN ===\n")
    
    # Get a job description
    jobs = load_sample_jobs(1)
    job = jobs[0]
    
    print(f"Job: {job['title']} at {job['companyName']}\n")
    print("Running chain: Extract → Categorize → Learning Path\n")
    print("=" * 50)
    
    result = sequential_chain(job["description"])
    
    print("\n" + "=" * 50)
    print("\n=== FINAL OUTPUT: Learning Path ===")
    print(result["learning_path"])
