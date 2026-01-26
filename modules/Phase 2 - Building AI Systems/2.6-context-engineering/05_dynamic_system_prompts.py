"""
05 - Dynamic System Prompts
===========================
Adjust system prompt based on task or context.

Key concept: System prompts can be templates filled with runtime data.

Book reference: AI_eng.5
"""

from openai import OpenAI
from datetime import datetime

client = OpenAI()


def create_system_prompt(
    user_profile: dict = None,
    task_type: str = "general",
    context_data: dict = None,
) -> str:
    """Create a dynamic system prompt based on context."""
    
    # Base role
    prompt_parts = ["You are a professional job search assistant."]
    
    # Add user personalization
    if user_profile:
        if user_profile.get("name"):
            prompt_parts.append(f"You are helping {user_profile['name']}.")
        if user_profile.get("experience_level"):
            prompt_parts.append(f"They are at {user_profile['experience_level']} level.")
        if user_profile.get("preferences"):
            prompt_parts.append(f"Their preferences: {', '.join(user_profile['preferences'])}.")
    
    # Add task-specific instructions
    task_instructions = {
        "search": "Focus on finding relevant job matches. List key requirements.",
        "compare": "Provide balanced comparisons. Use a structured format.",
        "summarize": "Be concise. Highlight the most important points.",
        "explain": "Be educational. Use examples when helpful.",
        "apply": "Be encouraging and practical. Focus on next steps.",
    }
    
    if task_type in task_instructions:
        prompt_parts.append(f"\nTask: {task_instructions[task_type]}")
    
    # Add context data
    if context_data:
        if context_data.get("job_count"):
            prompt_parts.append(f"\nDatabase has {context_data['job_count']} jobs available.")
        if context_data.get("last_search"):
            prompt_parts.append(f"User's last search: {context_data['last_search']}")
    
    # Add current date for relevance
    prompt_parts.append(f"\nCurrent date: {datetime.now().strftime('%B %d, %Y')}")
    
    return "\n".join(prompt_parts)


def query_with_dynamic_prompt(
    user_query: str,
    user_profile: dict = None,
    task_type: str = "general",
    context_data: dict = None,
) -> tuple[str, str]:
    """Query with a dynamically generated system prompt."""
    
    system_prompt = create_system_prompt(user_profile, task_type, context_data)
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.output_text, system_prompt


if __name__ == "__main__":
    print("=== DYNAMIC SYSTEM PROMPTS ===\n")
    
    # Different user profiles
    profiles = [
        {"name": "Alex", "experience_level": "junior", "preferences": ["remote", "startup"]},
        {"name": "Jordan", "experience_level": "senior", "preferences": ["leadership", "enterprise"]},
        None,  # Anonymous user
    ]
    
    task_types = ["search", "compare", "explain"]
    
    # Demo with first profile and search task
    profile = profiles[0]
    context = {"job_count": 10342, "last_search": "Python developer"}
    
    print("=== GENERATED PROMPT ===")
    _, prompt = query_with_dynamic_prompt(
        "What jobs match my skills?",
        user_profile=profile,
        task_type="search",
        context_data=context
    )
    print(prompt)
    
    print("\n=== DIFFERENT TASK TYPES ===\n")
    
    for task in task_types:
        response, _ = query_with_dynamic_prompt(
            "Tell me about Python developer jobs",
            user_profile=profile,
            task_type=task,
        )
        print(f"Task: {task}")
        print(f"Response: {response[:150]}...")
        print()
