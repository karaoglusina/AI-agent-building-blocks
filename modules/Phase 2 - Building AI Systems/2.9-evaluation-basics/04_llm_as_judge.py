"""
04 - LLM as Judge
=================
Use LLM to evaluate outputs.

Key concept: LLMs can evaluate quality, relevance, and correctness at scale.

Book reference: AI_eng.3, hands_on_LLM.III.12
"""

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class QualityScore(BaseModel):
    """Evaluation of a single response."""
    relevance: int = Field(ge=1, le=5, description="How relevant to the question (1-5)")
    accuracy: int = Field(ge=1, le=5, description="How factually correct (1-5)")
    helpfulness: int = Field(ge=1, le=5, description="How helpful to the user (1-5)")
    reasoning: str = Field(description="Brief explanation of scores")


class PairwiseJudgment(BaseModel):
    """Comparison of two responses."""
    winner: str = Field(description="'A', 'B', or 'tie'")
    reasoning: str


def evaluate_quality(question: str, response: str) -> QualityScore:
    """Evaluate a response's quality using LLM-as-judge."""
    result = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Evaluate the quality of this response to the given question. "
                           "Score 1-5 for relevance, accuracy, and helpfulness."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nResponse: {response}"
            }
        ],
        text_format=QualityScore
    )
    return result.output_parsed


def pairwise_comparison(question: str, response_a: str, response_b: str) -> PairwiseJudgment:
    """Compare two responses and pick a winner."""
    result = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Compare these two responses. Pick 'A', 'B', or 'tie' and explain why."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nResponse A: {response_a}\n\nResponse B: {response_b}"
            }
        ],
        text_format=PairwiseJudgment
    )
    return result.output_parsed


# Test cases
QUESTION = "What skills are most important for a Python developer?"

RESPONSES = [
    # Good response
    "Key skills for Python developers include: 1) Core Python proficiency (data structures, OOP), "
    "2) Web frameworks (Django/FastAPI), 3) Database knowledge (SQL, PostgreSQL), "
    "4) Version control (Git), and 5) Testing practices.",
    
    # Mediocre response
    "Python developers should know Python and how to code. They also need to be good at computers.",
    
    # Off-topic response
    "JavaScript is the most popular programming language for web development. "
    "React and Vue are common frameworks.",
]


if __name__ == "__main__":
    print("=== LLM AS JUDGE ===\n")
    print(f"Question: {QUESTION}\n")
    
    # Evaluate each response
    print("--- Quality Scores ---")
    for i, response in enumerate(RESPONSES, 1):
        print(f"\nResponse {i}: {response[:80]}...")
        
        score = evaluate_quality(QUESTION, response)
        print(f"  Relevance: {score.relevance}/5")
        print(f"  Accuracy: {score.accuracy}/5")
        print(f"  Helpfulness: {score.helpfulness}/5")
        print(f"  Avg: {(score.relevance + score.accuracy + score.helpfulness) / 3:.1f}")
        print(f"  Reasoning: {score.reasoning}")
    
    # Pairwise comparison
    print("\n--- Pairwise Comparison ---")
    print(f"\nComparing Response 1 vs Response 2:")
    
    comparison = pairwise_comparison(QUESTION, RESPONSES[0], RESPONSES[1])
    print(f"  Winner: Response {comparison.winner}")
    print(f"  Reasoning: {comparison.reasoning}")
    
    print("\n=== LLM-AS-JUDGE TIPS ===")
    print("• Use a stronger model as judge (GPT-4 > GPT-3.5)")
    print("• Randomize A/B order to avoid position bias")
    print("• Use specific rubrics for consistent scoring")
    print("• Validate against human judgments")
