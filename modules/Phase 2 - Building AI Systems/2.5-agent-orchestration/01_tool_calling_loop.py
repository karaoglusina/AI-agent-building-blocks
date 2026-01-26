"""
01 - Tool Calling Loop
======================
Basic agent loop: think → act → observe.

Key concept: Agents are LLMs in a loop that can call tools and observe results.

Book reference: AI_eng.6, hands_on_LLM.II.7, NLP_cook.10
"""

import json
from openai import OpenAI

client = OpenAI()

# Define available tools
TOOLS = [
    {
        "type": "function",
        "name": "search_jobs",
        "description": "Search for job postings by keyword",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Search keyword"}
            },
            "required": ["keyword"]
        }
    },
    {
        "type": "function",
        "name": "get_job_count",
        "description": "Get total number of jobs matching criteria",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Job category"}
            },
            "required": ["category"]
        }
    }
]


def search_jobs(keyword: str) -> str:
    """Mock job search function."""
    return json.dumps([
        {"title": f"Senior {keyword} Developer", "company": "TechCorp"},
        {"title": f"{keyword} Engineer", "company": "StartupXYZ"},
    ])


def get_job_count(category: str) -> str:
    """Mock job count function."""
    counts = {"engineering": 150, "data": 75, "product": 30}
    return str(counts.get(category.lower(), 10))


FUNCTION_MAP = {
    "search_jobs": search_jobs,
    "get_job_count": get_job_count,
}


def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """Run the agent loop: think → act → observe."""
    messages = [
        {"role": "system", "content": "You help users find jobs. Use tools when needed."},
        {"role": "user", "content": user_query}
    ]
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")
        
        # Think: Get LLM response
        response = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            tools=TOOLS
        )
        
        # Check if done (no tool calls)
        if not response.output:
            return response.output_text
        
        # Check for tool calls
        tool_calls = [item for item in response.output if item.type == "function_call"]
        
        if not tool_calls:
            return response.output_text
        
        # Act: Execute tool calls
        for tool_call in tool_calls:
            func_name = tool_call.name
            func_args = json.loads(tool_call.arguments)
            
            print(f"Calling: {func_name}({func_args})")
            
            # Execute the function
            result = FUNCTION_MAP[func_name](**func_args)
            
            print(f"Result: {result}")
            
            # Observe: Add result to messages
            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": result
            })
    
    return "Max iterations reached"


if __name__ == "__main__":
    print("=== TOOL CALLING AGENT ===")
    
    queries = [
        "Find Python developer jobs",
        "How many engineering jobs are there?",
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        result = run_agent(query)
        print(f"\nAgent: {result}")
        print("=" * 50)
