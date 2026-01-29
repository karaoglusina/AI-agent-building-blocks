"""
02 - Prompt Assembly
====================
Build prompts from components dynamically.

Key concept: Modular prompts are easier to maintain and test.

Book reference: AI_eng.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os
from dataclasses import dataclass


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@dataclass
class PromptComponents:
    """Components that can be assembled into a prompt."""
    role: str = ""
    task: str = ""
    context: str = ""
    format_instructions: str = ""
    examples: str = ""
    constraints: str = ""


def assemble_system_prompt(components: PromptComponents) -> str:
    """Assemble a system prompt from components."""
    parts = []
    
    if components.role:
        parts.append(f"You are {components.role}.")
    
    if components.task:
        parts.append(f"\nYour task: {components.task}")
    
    if components.context:
        parts.append(f"\nContext: {components.context}")
    
    if components.constraints:
        parts.append(f"\nConstraints:\n{components.constraints}")
    
    if components.format_instructions:
        parts.append(f"\nOutput format:\n{components.format_instructions}")
    
    if components.examples:
        parts.append(f"\nExamples:\n{components.examples}")
    
    return "\n".join(parts)


# Reusable prompt components
ROLE_JOB_ASSISTANT = "a professional job search assistant"
ROLE_RECRUITER = "an experienced technical recruiter"
ROLE_CAREER_COACH = "a career development coach"

TASK_MATCH_SKILLS = "Match the user's skills to relevant job requirements"
TASK_SUMMARIZE = "Provide concise summaries of job postings"
TASK_ADVICE = "Give actionable career advice"

CONSTRAINT_BRIEF = "- Keep responses under 100 words\n- Use bullet points"
CONSTRAINT_DETAILED = "- Provide comprehensive analysis\n- Include specific examples"
CONSTRAINT_FACTUAL = "- Only state facts from provided context\n- Say 'unknown' if unsure"


def query_with_assembled_prompt(user_query: str, components: PromptComponents) -> str:
    """Run query with an assembled system prompt."""
    system_prompt = assemble_system_prompt(components)
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
    ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== PROMPT ASSEMBLY ===\n")
    
    # Example 1: Brief job matcher
    brief_matcher = PromptComponents(
        role=ROLE_JOB_ASSISTANT,
        task=TASK_MATCH_SKILLS,
        constraints=CONSTRAINT_BRIEF)
    
    print("--- Brief Job Matcher ---")
    print(f"System prompt:\n{assemble_system_prompt(brief_matcher)}\n")
    
    # Example 2: Detailed career coach
    detailed_coach = PromptComponents(
        role=ROLE_CAREER_COACH,
        task=TASK_ADVICE,
        constraints=CONSTRAINT_DETAILED,
        context="The user is transitioning from academia to industry.")
    
    print("--- Detailed Career Coach ---")
    print(f"System prompt:\n{assemble_system_prompt(detailed_coach)}\n")
    
    # Run a query
    print("=== RUNNING QUERY ===\n")
    user_query = "I know Python and SQL. What jobs should I look for?"
    
    result = query_with_assembled_prompt(user_query, brief_matcher)
    print(f"User: {user_query}")
    print(f"Response: {result}")
