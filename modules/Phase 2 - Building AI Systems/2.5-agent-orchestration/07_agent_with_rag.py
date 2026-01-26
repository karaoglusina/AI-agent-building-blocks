"""
07 - Agent with RAG Tool
========================
Agent that can search a knowledge base.

Key concept: RAG as a tool lets the agent decide when to retrieve information.

Book reference: AI_eng.6, hands_on_LLM.II.8
"""

import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

# Setup
client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="job_kb", embedding_function=openai_ef)

# Index jobs
jobs = load_sample_jobs(30)
for job in jobs:
    collection.add(
        documents=[f"{job['title']} at {job['companyName']}\n{job['description'][:400]}"],
        ids=[job["id"]],
        metadatas=[{"title": job["title"], "company": job["companyName"]}]
    )


def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """Search the job knowledge base."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return json.dumps([
        {"title": m["title"], "company": m["company"], "content": d[:200]}
        for m, d in zip(results["metadatas"][0], results["documents"][0])
    ])


TOOLS = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Search the job database for relevant positions. Use for questions about specific jobs, skills, or companies.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {"type": "integer", "description": "Number of results", "default": 3}
            },
            "required": ["query"]
        }
    }
]


def run_rag_agent(user_query: str) -> str:
    """Run agent with RAG capability."""
    messages = [
        {
            "role": "system",
            "content": "You're a job search assistant. Use the knowledge base to answer questions. "
                       "If the answer isn't in the search results, say so."
        },
        {"role": "user", "content": user_query}
    ]
    
    response = client.responses.create(model="gpt-4o-mini", input=messages, tools=TOOLS)
    
    # Process tool calls
    for item in response.output:
        if item.type == "function_call" and item.name == "search_knowledge_base":
            args = json.loads(item.arguments)
            print(f"  Searching KB: \"{args['query']}\"")
            
            result = search_knowledge_base(args["query"], args.get("n_results", 3))
            parsed = json.loads(result)
            print(f"  Found {len(parsed)} results")
            
            messages.append({"role": "assistant", "content": None, "tool_calls": [item]})
            messages.append({"role": "tool", "tool_call_id": item.call_id, "content": result})
    
    # Generate final answer
    final = client.responses.create(model="gpt-4o-mini", input=messages)
    return final.output_text


if __name__ == "__main__":
    print(f"=== AGENT WITH RAG ({collection.count()} jobs indexed) ===\n")
    
    queries = [
        "What Python developer jobs are available?",
        "Which companies are hiring for data roles?",
        "Tell me about remote engineering positions",
        "What's the weather like today?",  # Not in KB
    ]
    
    for query in queries:
        print(f"User: {query}")
        answer = run_rag_agent(query)
        print(f"Agent: {answer}\n")
        print("-" * 50 + "\n")
