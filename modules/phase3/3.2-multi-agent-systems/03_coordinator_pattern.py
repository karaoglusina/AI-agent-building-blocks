"""
03 - Coordinator Pattern
=========================
Orchestrator agent that delegates tasks to specialized worker agents.

Key concept: Central coordinator decomposes tasks and routes to appropriate agents.

Book reference: AI_eng.6, AI_eng.10.5
"""

import json
from openai import OpenAI
from pydantic import BaseModel

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()


class TaskPlan(BaseModel):
    """Plan for task decomposition."""
    subtasks: list[str]
    agent_assignments: list[str]  # Which agent handles each subtask


class WorkerAgent:
    """Specialized worker agent."""

    def __init__(self, name: str, capability: str):
        self.name = name
        self.capability = capability

    def execute(self, task: str) -> str:
        """Execute assigned task."""
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": f"You are a {self.capability}. Execute the assigned task concisely."},
                {"role": "user", "content": task}
            ]
        )
        return response.output_text


class Coordinator:
    """Orchestrator that delegates to workers."""

    def __init__(self, workers: dict[str, WorkerAgent]):
        self.workers = workers

    def plan_tasks(self, user_query: str) -> TaskPlan:
        """Decompose user query into subtasks."""
        available_agents = "\n".join([f"- {name}: {agent.capability}" for name, agent in self.workers.items()])

        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": f"You are a task coordinator. Break down the user query into subtasks.\n"
                               f"Available agents:\n{available_agents}\n\n"
                               f"Assign each subtask to an agent name from the list above."
                },
                {"role": "user", "content": user_query}
            ],
            text_format=TaskPlan
        )
        return response.output_parsed

    def execute(self, user_query: str) -> str:
        """Coordinate execution of user query."""
        print(f"\n[Coordinator] Planning tasks for: {user_query}")

        # Plan and decompose
        plan = self.plan_tasks(user_query)

        results = []
        for i, (subtask, agent_name) in enumerate(zip(plan.subtasks, plan.agent_assignments), 1):
            print(f"\n[Coordinator] Subtask {i}: {subtask}")
            print(f"[Coordinator] Assigning to: {agent_name}")

            worker = self.workers.get(agent_name)
            if not worker:
                results.append(f"Error: Agent '{agent_name}' not found")
                continue

            result = worker.execute(subtask)
            print(f"[{worker.name}] Result: {result[:100]}...")
            results.append(result)

        # Synthesize final answer
        synthesis_prompt = f"""
User query: {user_query}

Results from worker agents:
{chr(10).join([f"{i+1}. {r}" for i, r in enumerate(results)])}

Synthesize a coherent final answer to the user query.
"""

        final = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": "You synthesize information from multiple sources into a coherent answer."},
                {"role": "user", "content": synthesis_prompt}
            ]
        )

        return final.output_text


if __name__ == "__main__":
    print("=== COORDINATOR PATTERN ===")

    # Define worker agents
    workers = {
        "search_agent": WorkerAgent("Search Agent", "job search specialist who finds relevant positions"),
        "analysis_agent": WorkerAgent("Analysis Agent", "job market analyst who provides insights and trends"),
        "recommendation_agent": WorkerAgent("Recommendation Agent", "career advisor who provides personalized recommendations"),
    }

    coordinator = Coordinator(workers)

    queries = [
        "Find Python jobs and tell me which companies are hiring most",
        "What are the top skills for data science roles and recommend good companies",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        result = coordinator.execute(query)
        print(f"\n[Final Answer]: {result}")
        print("\n" + "=" * 50)
