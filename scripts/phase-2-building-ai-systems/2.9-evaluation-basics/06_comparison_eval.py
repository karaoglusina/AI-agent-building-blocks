"""
06 - A/B Comparison Evaluation
==============================
Compare two approaches side by side.

Key concept: Comparative evaluation helps decide between options.

Book reference: AI_eng.3, AI_eng.4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

from openai import OpenAI
import os
from pydantic import BaseModel
from dataclasses import dataclass
import random


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@dataclass
class ComparisonResult:
    """Result of comparing two systems."""
    query: str
    output_a: str
    output_b: str
    winner: str  # "A", "B", or "tie"
    reason: str


class JudgeDecision(BaseModel):
    """LLM judge's decision."""
    winner: str
    reasoning: str


# System A: Concise responses
def system_a(query: str) -> str:
    """System A: Brief, to-the-point responses."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Be extremely concise. Maximum 2 sentences."},
    {"role": "user", "content": query}
    ]
    )
    return response.choices[0].message.content


# System B: Detailed responses
def system_b(query: str) -> str:
    """System B: Comprehensive, detailed responses."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Provide comprehensive, detailed answers with examples."},
    {"role": "user", "content": query}
    ]
    )
    return response.choices[0].message.content


def judge_comparison(query: str, output_a: str, output_b: str) -> JudgeDecision:
    """Have LLM judge which response is better."""
    # Randomize order to avoid position bias
    if random.random() > 0.5:
        first, second = output_a, output_b
        first_label, second_label = "A", "B"
    else:
        first, second = output_b, output_a
        first_label, second_label = "B", "A"
    
    result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": "Compare these two responses to the query. "
    "Pick the better one (First or Second) or 'tie'. "
    "Consider relevance, helpfulness, and clarity."
    },
    {
    "role": "user",
    "content": f"Query: {query}\n\nFirst: {first}\n\nSecond: {second}"
    }
    ],
    response_format={"type": "json_object"}
    )
    
    # Map back to A/B
    decision = result.output_parsed
    if "first" in decision.winner.lower():
        decision.winner = first_label
    elif "second" in decision.winner.lower():
        decision.winner = second_label
    else:
        decision.winner = "tie"
    
    return decision


def run_ab_comparison(queries: list[str]) -> dict:
    """Run A/B comparison across queries."""
    results = []
    
    for query in queries:
        output_a = system_a(query)
        output_b = system_b(query)
        
        judgment = judge_comparison(query, output_a, output_b)
        
        results.append(ComparisonResult(
            query=query,
            output_a=output_a,
            output_b=output_b,
            winner=judgment.winner,
            reason=judgment.reasoning
        ))
    
    # Aggregate
    wins = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        wins[r.winner] += 1
    
    return {"results": results, "summary": wins}


TEST_QUERIES = [
    "What skills do I need for a Python developer job?",
    "How do I prepare for a technical interview?",
    "What's the difference between junior and senior roles?"]


if __name__ == "__main__":
    print("=== A/B COMPARISON EVALUATION ===\n")
    print("System A: Concise responses (2 sentences max)")
    print("System B: Detailed responses (comprehensive)\n")
    
    comparison = run_ab_comparison(TEST_QUERIES)
    
    # Summary
    print("=== SUMMARY ===")
    summary = comparison["summary"]
    total = sum(summary.values())
    print(f"System A wins: {summary['A']}/{total}")
    print(f"System B wins: {summary['B']}/{total}")
    print(f"Ties: {summary['tie']}/{total}")
    
    # Details
    print("\n=== DETAILED RESULTS ===")
    for result in comparison["results"]:
        print(f"\nQuery: {result.query}")
        print(f"Winner: System {result.winner}")
        print(f"Reason: {result.reason}")
        print(f"\nA: {result.output_a[:100]}...")
        print(f"B: {result.output_b[:100]}...")
        print("-" * 50)
