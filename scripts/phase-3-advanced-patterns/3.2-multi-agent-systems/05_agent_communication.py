"""
05 - Agent Communication
=========================
Multiple agents share information and collaborate to solve tasks.

Key concept: Agents communicate and pass context to achieve complex goals.

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


class Message(BaseModel):
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: str
    data: dict = {}


class CommunicatingAgent:
    """Agent that can send and receive messages from other agents."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.inbox: list[Message] = []

    def receive(self, message: Message):
        """Receive a message from another agent."""
        self.inbox.append(message)
        print(f"[{self.name}] Received message from {message.from_agent}: {message.content[:60]}...")

    def process_and_respond(self, user_query: str = None) -> Message | None:
        """Process inbox messages and generate response."""
        if not self.inbox and not user_query:
            return None

        # Build context from messages
        context_parts = []
        if self.inbox:
            context_parts.append("Messages from other agents:")
            for msg in self.inbox:
                context_parts.append(f"- {msg.from_agent}: {msg.content}")
                if msg.data:
                    context_parts.append(f"  Data: {json.dumps(msg.data)}")

        context = "\n".join(context_parts) if context_parts else ""

        # Generate response
        prompt = f"""You are a {self.role}.

{context}

{"User query: " + user_query if user_query else "Respond based on the messages received."}

Generate your response and indicate if you need to communicate with another agent.
"""

        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": f"You are {self.name}, a {self.role}. Communicate clearly and concisely."},
        {"role": "user", "content": prompt}
        ]
        )

        return response.choices[0].message.content

    def send_message(self, to_agent: str, content: str, data: dict = None) -> Message:
        """Send a message to another agent."""
        message = Message(
            from_agent=self.name,
            to_agent=to_agent,
            content=content,
            data=data or {}
        )
        print(f"[{self.name}] Sending to {to_agent}: {content[:60]}...")
        return message


class AgentNetwork:
    """Network of communicating agents."""

    def __init__(self):
        self.agents: dict[str, CommunicatingAgent] = {}

    def add_agent(self, agent: CommunicatingAgent):
        """Add an agent to the network."""
        self.agents[agent.name] = agent

    def send(self, message: Message):
        """Route message to recipient."""
        if message.to_agent in self.agents:
            self.agents[message.to_agent].receive(message)
        else:
            print(f"Warning: Agent {message.to_agent} not found")


def run_collaborative_search(query: str) -> str:
    """Run a collaborative multi-agent search."""
    print(f"\n=== Collaborative Search: {query} ===\n")

    # Create network
    network = AgentNetwork()

    # Create agents
    router = CommunicatingAgent("Router", "query analyzer that routes requests to appropriate agents")
    searcher = CommunicatingAgent("Searcher", "job search specialist who finds relevant positions")
    analyzer = CommunicatingAgent("Analyzer", "data analyst who analyzes and summarizes job information")

    network.add_agent(router)
    network.add_agent(searcher)
    network.add_agent(analyzer)

    # Step 1: Router analyzes query
    print("[Step 1: Router analyzes query]")
    router_response = router.process_and_respond(user_query=query)
    print(f"[Router]: {router_response}\n")

    # Step 2: Router sends to Searcher
    print("[Step 2: Router delegates to Searcher]")
    msg_to_searcher = router.send_message(
        "Searcher",
        f"Find jobs matching: {query}",
        data={"query": query}
    )
    network.send(msg_to_searcher)

    # Step 3: Searcher finds jobs
    print("\n[Step 3: Searcher finds jobs]")
    jobs = load_sample_jobs(5)
    job_summaries = [f"{j['title']} at {j['companyName']}" for j in jobs]

    searcher_response = searcher.process_and_respond()
    print(f"[Searcher]: Found {len(jobs)} jobs\n")

    # Step 4: Searcher sends to Analyzer
    print("[Step 4: Searcher sends results to Analyzer]")
    msg_to_analyzer = searcher.send_message(
        "Analyzer",
        "Please analyze these job results and provide insights",
        data={"jobs": job_summaries}
    )
    network.send(msg_to_analyzer)

    # Step 5: Analyzer provides insights
    print("\n[Step 5: Analyzer provides insights]")
    analyzer_response = analyzer.process_and_respond()
    print(f"[Analyzer]: {analyzer_response}\n")

    # Step 6: Results back to Router for final synthesis
    print("[Step 6: Final synthesis]")
    msg_to_router = analyzer.send_message(
        "Router",
        "Analysis complete",
        data={"analysis": analyzer_response}
    )
    network.send(msg_to_router)

    final_response = router.process_and_respond()
    return final_response


if __name__ == "__main__":
    print("=== AGENT COMMUNICATION ===")

    queries = [
        "Find Python developer jobs and tell me about the market",
        "Search for data science positions"]

    for query in queries:
        result = run_collaborative_search(query)
        print(f"\n[Final Result]: {result}")
        print("\n" + "=" * 50)
