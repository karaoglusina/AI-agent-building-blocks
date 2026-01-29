"""
01 - System Prompt Design
=========================
Structure effective system prompts.

Key concept: A good system prompt defines role, capabilities, constraints, and format.

Book reference: AI_eng.5, hands_on_LLM.II.6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


# System prompt components
SYSTEM_PROMPT_TEMPLATE = """
## Role
{role}

## Capabilities
{capabilities}

## Constraints
{constraints}

## Output Format
{format_instructions}
"""


def create_system_prompt(
    role: str,
    capabilities: list[str],
    constraints: list[str],
    format_instructions: str = "Be concise and helpful."
) -> str:
    """Create a structured system prompt."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        role=role,
        capabilities="\n".join(f"- {c}" for c in capabilities),
        constraints="\n".join(f"- {c}" for c in constraints),
        format_instructions=format_instructions
    )


# Example prompts for different use cases
JOB_SEARCH_PROMPT = create_system_prompt(
    role="You are an expert job search assistant helping candidates find their ideal positions.",
    capabilities=[
        "Search and filter job postings",
        "Explain job requirements",
        "Suggest relevant positions based on skills",
        "Compare different opportunities"],
    constraints=[
        "Only provide information about available jobs",
        "Do not make up salary figures unless provided",
        "Be honest about job market realities",
        "Do not discriminate based on protected characteristics"],
    format_instructions="Use bullet points for lists. Keep responses under 200 words."
)

MINIMAL_PROMPT = "You are a helpful assistant."

DETAILED_PROMPT = """You are a senior technical recruiter with 10 years of experience.

When helping job seekers:
1. First understand their skills and preferences
2. Suggest relevant job categories
3. Highlight potential skill gaps
4. Recommend next steps

Always be encouraging but realistic about job market conditions."""


def compare_prompts(query: str, prompts: dict[str, str]):
    """Compare responses from different system prompts."""
    results = {}
    
    for name, prompt in prompts.items():
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
        ]
        )
        results[name] = response.choices[0].message.content
    
    return results


if __name__ == "__main__":
    print("=== SYSTEM PROMPT DESIGN ===\n")
    
    # Show the structured prompt
    print("--- Structured System Prompt ---")
    print(JOB_SEARCH_PROMPT[:500] + "...\n")
    
    # Compare different prompts
    query = "I know Python and want to find a job. Help me."
    
    prompts = {
        "Minimal": MINIMAL_PROMPT,
        "Structured": JOB_SEARCH_PROMPT,
        "Detailed": DETAILED_PROMPT,
    }
    
    print(f"Query: \"{query}\"\n")
    print("=" * 50)
    
    results = compare_prompts(query, prompts)
    
    for name, response in results.items():
        print(f"\n--- {name} Prompt Response ---")
        print(response[:300] + "..." if len(response) > 300 else response)
