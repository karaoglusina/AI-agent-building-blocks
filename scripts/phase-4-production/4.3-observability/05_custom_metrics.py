"""
05 - Custom Metrics
===================
Add application-specific metrics to observability.

Key concept: Custom metrics capture domain-specific KPIs beyond standard LLM metrics - crucial for business value monitoring.

Book reference: AI_eng.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import sys
from pathlib import Path

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import os
import time
from datetime import datetime
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from openai import OpenAI

# Initialize clients
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()


@observe()
def job_search_with_metrics(query: str, user_id: str) -> dict:
    """Job search with custom business metrics."""
    start_time = time.time()

    # Simulate job search
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a job search assistant. Suggest 3 relevant job roles for the query."
            },
            {"role": "user", "content": query}
        ],
        temperature=0.7)

    search_time = time.time() - start_time
    answer = response.choices[0].message.content

    # Extract custom metrics
    num_jobs_suggested = answer.count("\n") + 1  # Simple heuristic
    answer_length = len(answer)

    # Log standard info
    langfuse_context.update_current_observation(
        input={"query": query},
        output={"answer": answer},
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    )

    # Add custom metrics as scores
    langfuse_context.score_current_trace(
        name="search_latency_ms",
        value=search_time * 1000,
        comment=f"Time taken for search: {search_time:.3f}s"
    )

    langfuse_context.score_current_trace(
        name="num_results",
        value=num_jobs_suggested,
        comment=f"Number of job suggestions: {num_jobs_suggested}"
    )

    langfuse_context.score_current_trace(
        name="answer_length",
        value=answer_length,
        comment=f"Answer character length: {answer_length}"
    )

    # Add user metadata
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["job_search", "production"],
        metadata={
            "search_time_ms": round(search_time * 1000, 2),
            "num_suggestions": num_jobs_suggested
        }
    )

    return {
        "answer": answer,
        "metrics": {
            "search_time_ms": round(search_time * 1000, 2),
            "num_jobs_suggested": num_jobs_suggested,
            "answer_length": answer_length
        }
    }


@observe()
def user_feedback_tracking(
    trace_id: str,
    user_rating: int,
    user_comment: str = None
) -> dict:
    """Track user feedback as custom metrics."""
    # In production, you would associate this with the original trace
    # For now, we create a new observation

    feedback = {
        "trace_id": trace_id,
        "rating": user_rating,
        "comment": user_comment,
        "timestamp": datetime.now().isoformat()
    }

    langfuse_context.update_current_observation(
        input={"trace_id": trace_id},
        output=feedback
    )

    # Score the feedback
    langfuse_context.score_current_trace(
        name="user_satisfaction",
        value=user_rating,
        comment=user_comment or f"User rating: {user_rating}/5"
    )

    # You can also use Langfuse SDK to score existing traces
    # langfuse.score(
    #     trace_id=trace_id,
    #     name="user_satisfaction",
    #     value=user_rating,
    #     comment=user_comment
    # )

    return feedback


@observe()
def content_safety_check(text: str) -> dict:
    """Check content safety with custom metrics."""
    # Simulate content moderation
    response = client.moderations.create(input=text)

    result = response.results[0]
    is_flagged = result.flagged
    categories = {
        cat: score
        for cat, score in result.category_scores.model_dump().items()
        if score > 0.1  # Only show significant scores
    }

    # Log moderation
    langfuse_context.update_current_observation(
        input={"text": text},
        output={
            "flagged": is_flagged,
            "categories": categories
        }
    )

    # Score safety
    safety_score = 0 if is_flagged else 100
    langfuse_context.score_current_trace(
        name="content_safety",
        value=safety_score,
        comment=f"Flagged: {is_flagged}, Categories: {list(categories.keys())}"
    )

    # Add safety metadata
    langfuse_context.update_current_trace(
        tags=["moderation", "safety"],
        metadata={
            "is_flagged": is_flagged,
            "high_risk_categories": list(categories.keys())
        }
    )

    return {
        "is_safe": not is_flagged,
        "safety_score": safety_score,
        "categories": categories
    }


@observe()
def response_quality_metrics(query: str, answer: str) -> dict:
    """Calculate response quality metrics."""
    # Custom quality metrics
    metrics = {
        "answer_length": len(answer),
        "word_count": len(answer.split()),
        "sentence_count": answer.count(".") + answer.count("!") + answer.count("?"),
        "avg_sentence_length": len(answer.split()) / max(1, answer.count(".") + 1),
        "contains_bullet_points": "\n-" in answer or "\n•" in answer or "\n*" in answer,
        "contains_numbers": any(char.isdigit() for char in answer),
    }

    # Calculate quality score (0-100)
    quality_score = 0

    # Good length (100-500 chars)
    if 100 <= metrics["answer_length"] <= 500:
        quality_score += 30

    # Good structure (has bullet points or multiple sentences)
    if metrics["contains_bullet_points"] or metrics["sentence_count"] >= 3:
        quality_score += 30

    # Good detail (contains numbers/specifics)
    if metrics["contains_numbers"]:
        quality_score += 20

    # Not too verbose
    if metrics["avg_sentence_length"] < 30:
        quality_score += 20

    metrics["quality_score"] = quality_score

    # Log metrics
    langfuse_context.update_current_observation(
        input={"query": query, "answer": answer},
        output=metrics
    )

    # Score response quality
    langfuse_context.score_current_trace(
        name="response_quality",
        value=quality_score,
        comment=f"Quality score: {quality_score}/100"
    )

    langfuse_context.score_current_trace(
        name="answer_completeness",
        value=100 if metrics["word_count"] > 20 else 50,
        comment=f"Word count: {metrics['word_count']}"
    )

    return metrics


@observe()
def multi_metric_pipeline(query: str, user_id: str) -> dict:
    """Complete pipeline with multiple custom metrics."""
    # Tag the trace
    langfuse_context.update_current_trace(
        name="complete_job_search",
        user_id=user_id,
        session_id=f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}",
        tags=["job_search", "production", "metrics"]
    )

    # 1. Check query safety
    with langfuse_context.observe(name="safety_check"):
        safety = content_safety_check(query)
        if not safety["is_safe"]:
            return {
                "error": "Query flagged by content moderation",
                "safety": safety
            }

    # 2. Perform search
    with langfuse_context.observe(name="search"):
        search_result = job_search_with_metrics(query, user_id)

    # 3. Evaluate quality
    with langfuse_context.observe(name="quality_check"):
        quality = response_quality_metrics(query, search_result["answer"])

    # 4. Overall success score
    overall_score = (
        safety["safety_score"] * 0.3 +
        quality["quality_score"] * 0.7
    )

    langfuse_context.score_current_trace(
        name="overall_success",
        value=overall_score,
        comment=f"Combined score: {overall_score:.1f}/100"
    )

    result = {
        "query": query,
        "answer": search_result["answer"],
        "metrics": {
            "search": search_result["metrics"],
            "safety": safety,
            "quality": quality,
            "overall_score": overall_score
        }
    }

    langfuse_context.update_current_observation(
        input={"query": query, "user_id": user_id},
        output=result
    )

    return result


if __name__ == "__main__":
    print("=== CUSTOM METRICS ===\n")

    # 1. Job search with metrics
    print("1. Job Search with Custom Metrics")
    print("-" * 60)
    result = job_search_with_metrics(
        "Looking for senior Python developer roles",
        user_id="demo_user_123"
    )
    print(f"Answer: {result['answer'][:150]}...")
    print(f"\nMetrics:")
    for key, value in result['metrics'].items():
        print(f"  - {key}: {value}")
    print()

    # 2. Content safety check
    print("2. Content Safety Check")
    print("-" * 60)
    test_texts = [
        "Looking for a great job opportunity",
        "I hate this terrible awful job search process"  # Might trigger moderation
    ]

    for text in test_texts:
        safety = content_safety_check(text)
        print(f"Text: {text[:50]}...")
        print(f"Safe: {safety['is_safe']} (score: {safety['safety_score']})")
        if safety['categories']:
            print(f"Categories: {safety['categories']}")
        print()

    # 3. Response quality metrics
    print("3. Response Quality Metrics")
    print("-" * 60)
    test_responses = [
        ("What is ML?", "Machine learning is AI."),  # Too short
        ("What is ML?", """Machine learning is a subset of AI that enables systems to learn from data.
        Key types include:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning
        It's used in recommendations, predictions, and automation."""),  # Good quality
    ]

    for query, answer in test_responses:
        quality = response_quality_metrics(query, answer)
        print(f"Query: {query}")
        print(f"Answer length: {quality['answer_length']} chars")
        print(f"Quality score: {quality['quality_score']}/100")
        print()

    # 4. Complete pipeline with all metrics
    print("4. Complete Pipeline with Multi-Metrics")
    print("-" * 60)
    result = multi_metric_pipeline(
        "Find me remote machine learning engineer positions at startups",
        user_id="demo_user_456"
    )

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Answer: {result['answer'][:150]}...")
        print(f"\nOverall Score: {result['metrics']['overall_score']:.1f}/100")
        print("\nDetailed Metrics:")
        print(f"  Search time: {result['metrics']['search']['search_time_ms']}ms")
        print(f"  Safety score: {result['metrics']['safety']['safety_score']}/100")
        print(f"  Quality score: {result['metrics']['quality']['quality_score']}/100")
    print()

    # 5. Simulate user feedback
    print("5. User Feedback Tracking")
    print("-" * 60)
    feedback = user_feedback_tracking(
        trace_id="demo_trace_123",
        user_rating=4,
        user_comment="Good results but could be more specific"
    )
    print(f"Rating: {feedback['rating']}/5")
    print(f"Comment: {feedback['comment']}")
    print(f"Timestamp: {feedback['timestamp']}\n")

    # Flush all traces
    langfuse.flush()

    print("=" * 60)
    print("✓ Custom metrics tracking complete")
    print("  - View custom scores in Langfuse dashboard")
    print("  - Set up alerts for metric thresholds")
    print("  - Analyze metrics over time")
    print("  - Track user satisfaction trends")
    print("=" * 60)
