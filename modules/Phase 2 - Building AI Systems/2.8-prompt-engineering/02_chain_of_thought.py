"""
02 - Chain of Thought
=====================
Make the model reason step-by-step.

Key concept: Explicit reasoning steps improve accuracy on complex tasks.

Book reference: AI_eng.5, hands_on_LLM.II.6, speach_lang.I.12.4
"""

from openai import OpenAI

client = OpenAI()


def query_direct(question: str) -> str:
    """Ask question directly without CoT."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "Answer the question directly."},
            {"role": "user", "content": question}
        ]
    )
    return response.output_text


def query_with_cot(question: str) -> str:
    """Ask question with chain-of-thought prompting."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Think through this step by step. Show your reasoning before giving the final answer."
            },
            {"role": "user", "content": question}
        ]
    )
    return response.output_text


def query_with_structured_cot(question: str) -> str:
    """Ask with explicitly structured CoT."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": """Solve this problem using these steps:
1. UNDERSTAND: Restate the problem in your own words
2. PLAN: Outline your approach
3. EXECUTE: Work through each step
4. VERIFY: Check your answer
5. ANSWER: Give the final answer"""
            },
            {"role": "user", "content": question}
        ]
    )
    return response.output_text


# Test questions that benefit from reasoning
QUESTIONS = [
    # Reasoning question
    "A company has 150 job openings. 40% are for engineering, 30% are for sales, and the rest are for operations. They fill 80% of engineering roles, 60% of sales roles, and 50% of operations roles. How many total positions are filled?",
    
    # Job matching question
    "A candidate has 3 years of Python, 2 years of JavaScript, and 1 year of SQL. Job A requires 4+ years Python OR 3+ years JavaScript. Job B requires 2+ years Python AND 2+ years SQL. Which jobs are they qualified for?",
]


if __name__ == "__main__":
    print("=== CHAIN OF THOUGHT ===\n")
    
    for question in QUESTIONS:
        print(f"Question: {question}\n")
        print("=" * 50)
        
        # Direct answer
        print("\n--- Direct Answer ---")
        direct = query_direct(question)
        print(direct)
        
        # With CoT
        print("\n--- With Chain of Thought ---")
        cot = query_with_cot(question)
        print(cot)
        
        print("\n" + "=" * 50 + "\n")
    
    # Show structured CoT example
    print("=== STRUCTURED COT EXAMPLE ===\n")
    structured = query_with_structured_cot(QUESTIONS[0])
    print(structured)
