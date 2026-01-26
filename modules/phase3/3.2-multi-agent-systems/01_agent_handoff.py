"""
01 - Agent Handoff
==================
Transfer conversation between agents based on user needs.

Key concept: Different agents handle different conversation stages or domains.

Book reference: AI_eng.6
"""

import json
from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


class Agent:
    """Base agent with a specific role."""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def respond(self, messages: list[dict]) -> str:
        """Generate response for this agent."""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *messages
        ]

        response = client.responses.create(
            model="gpt-4o-mini",
            input=full_messages
        )
        return response.output_text


def classify_intent(user_input: str) -> str:
    """Classify user intent to determine which agent should handle it."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Classify the user's intent. Respond with only one word: "
                           "'search' (finding jobs), 'apply' (application help), or 'general' (other questions)."
            },
            {"role": "user", "content": user_input}
        ]
    )
    return response.output_text.strip().lower()


def run_multi_agent_handoff(user_input: str) -> tuple[str, str]:
    """Route user input to the appropriate agent."""
    # Define specialized agents
    agents = {
        "search": Agent(
            "Search Agent",
            "You help users search for jobs. Be concise and focus on job requirements. "
            "If the user wants to apply, tell them the Apply Agent can help with that."
        ),
        "apply": Agent(
            "Apply Agent",
            "You help users with job applications, resumes, and cover letters. "
            "If they want to search for jobs, tell them the Search Agent can help."
        ),
        "general": Agent(
            "General Agent",
            "You're a general assistant. Answer questions and route to specialized agents when needed."
        )
    }

    # Classify and route
    intent = classify_intent(user_input)
    agent = agents.get(intent, agents["general"])

    messages = [{"role": "user", "content": user_input}]
    response = agent.respond(messages)

    return agent.name, response


if __name__ == "__main__":
    print("=== AGENT HANDOFF ===\n")

    queries = [
        "I'm looking for Python developer jobs in Amsterdam",
        "How do I write a good cover letter?",
        "What's the weather like today?",
        "Can you help me search for data engineer positions?",
        "I need advice on my resume for a senior role",
    ]

    for query in queries:
        print(f"User: {query}")
        agent_name, response = run_multi_agent_handoff(query)
        print(f"[{agent_name}]: {response}\n")
        print("-" * 50 + "\n")
