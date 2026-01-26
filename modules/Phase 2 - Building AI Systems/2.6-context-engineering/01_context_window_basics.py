"""
01 - Context Window Basics
==========================
Token limits, counting, and efficiency considerations.

Key concept: Every model has a context window limit - know your budget.

Book reference: AI_eng.5, hands_on_LLM.I.3
"""

import tiktoken
from openai import OpenAI

client = OpenAI()

# Model context limits (as of 2024)
MODEL_LIMITS = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_message_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Count tokens in a message list (including overhead)."""
    encoding = tiktoken.encoding_for_model(model)
    
    # Each message has overhead tokens
    tokens = 0
    for msg in messages:
        tokens += 4  # Message overhead: <im_start>, role, \n, <im_end>
        tokens += len(encoding.encode(msg.get("content", "")))
        tokens += len(encoding.encode(msg.get("role", "")))
    
    tokens += 2  # Reply priming
    return tokens


def estimate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate cost for tokens (input pricing)."""
    # Prices per 1M tokens (approximate)
    prices = {
        "gpt-4o-mini": 0.15,
        "gpt-4o": 2.50,
        "gpt-4-turbo": 10.00,
    }
    return (tokens / 1_000_000) * prices.get(model, 0.15)


if __name__ == "__main__":
    print("=== CONTEXT WINDOW BASICS ===\n")
    
    # Show model limits
    print("Model context limits:")
    for model, limit in MODEL_LIMITS.items():
        print(f"  {model}: {limit:,} tokens")
    print()
    
    # Count tokens in sample text
    sample_text = """
    We are looking for a Senior Python Developer with 5+ years of experience.
    The ideal candidate will have expertise in Django, FastAPI, and PostgreSQL.
    Remote work is available for candidates in the EU timezone.
    """
    
    tokens = count_tokens(sample_text)
    print(f"Sample text: {len(sample_text)} chars, {tokens} tokens")
    print(f"  Ratio: ~{len(sample_text) / tokens:.1f} chars per token\n")
    
    # Count message tokens
    messages = [
        {"role": "system", "content": "You are a helpful job search assistant."},
        {"role": "user", "content": "Find me Python developer jobs in Amsterdam."},
        {"role": "assistant", "content": "I found 15 Python developer positions in Amsterdam."},
    ]
    
    msg_tokens = count_message_tokens(messages)
    print(f"Conversation ({len(messages)} messages): {msg_tokens} tokens")
    
    # Budget calculation
    model = "gpt-4o-mini"
    limit = MODEL_LIMITS[model]
    used_pct = (msg_tokens / limit) * 100
    print(f"  Using {used_pct:.3f}% of {model} context window")
    
    # Cost estimation
    cost = estimate_cost(msg_tokens)
    print(f"  Estimated cost: ${cost:.6f}\n")
    
    print("=== TOKEN EFFICIENCY TIPS ===")
    print("• Remove unnecessary whitespace")
    print("• Use abbreviations in system prompts")
    print("• Summarize old conversation turns")
    print("• Use shorter field names in structured output")
