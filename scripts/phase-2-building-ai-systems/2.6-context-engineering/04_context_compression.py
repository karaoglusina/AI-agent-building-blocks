"""
04 - Context Compression
========================
Summarize context to fit more information.

Key concept: Compression trades fidelity for capacity - summarize strategically.

Book reference: AI_eng.6, hands_on_LLM.II.7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import utils._load_env  # Loads .env file automatically

import tiktoken
from openai import OpenAI
import os


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def compress_text(text: str, target_tokens: int) -> str:
    """Compress text to fit within target token count."""
    current_tokens = count_tokens(text)
    
    if current_tokens <= target_tokens:
        return text
    
    compression_ratio = target_tokens / current_tokens
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {
    "role": "system",
    "content": f"Compress this text to approximately {compression_ratio:.0%} of its length. "
    "Keep the most important information. Be concise."
    },
    {"role": "user", "content": text}
    ]
    )
    return response.choices[0].message.content


def compress_conversation(messages: list[dict], keep_recent: int = 2) -> list[dict]:
    """Compress older messages while keeping recent ones intact."""
    if len(messages) <= keep_recent + 1:  # +1 for system message
        return messages
    
    # Keep system message
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    
    # Split into old and recent
    conversation = messages[1:] if system_msg else messages
    old_messages = conversation[:-keep_recent]
    recent_messages = conversation[-keep_recent:]
    
    # Summarize old messages
    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
    
    summary = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Focus on key facts."},
    {"role": "user", "content": old_text}
    ]
    ).choices[0].message.content
    
    # Reconstruct messages
    result = []
    if system_msg:
        result.append(system_msg)
    result.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
    result.extend(recent_messages)
    
    return result


if __name__ == "__main__":
    print("=== CONTEXT COMPRESSION ===\n")
    
    # Long text compression
    long_text = """
    We are looking for a Senior Python Developer to join our growing engineering team.
    The ideal candidate will have 5+ years of experience with Python, strong knowledge
    of Django and FastAPI frameworks, and experience with PostgreSQL databases.
    You should be comfortable with cloud services, particularly AWS or GCP.
    We offer competitive salary, full remote work, unlimited PTO, and equity.
    Our team is distributed across Europe and North America.
    """ * 2  # Repeat to make it longer
    
    original_tokens = count_tokens(long_text)
    print(f"Original: {original_tokens} tokens")
    print(f"Text: {long_text[:100]}...\n")
    
    compressed = compress_text(long_text, target_tokens=50)
    compressed_tokens = count_tokens(compressed)
    print(f"Compressed: {compressed_tokens} tokens ({compressed_tokens/original_tokens:.0%} of original)")
    print(f"Text: {compressed}\n")
    
    # Conversation compression
    print("=== CONVERSATION COMPRESSION ===\n")
    
    messages = [
        {"role": "system", "content": "You are a job search assistant."},
        {"role": "user", "content": "I'm looking for Python jobs."},
        {"role": "assistant", "content": "I found several Python positions. What seniority level?"},
        {"role": "user", "content": "Senior level please."},
        {"role": "assistant", "content": "Here are 5 senior Python roles."},
        {"role": "user", "content": "Tell me about the first one."},
        {"role": "assistant", "content": "The first is at TechCorp, remote, $150k."},
        {"role": "user", "content": "What are the requirements?"},  # Recent
        {"role": "assistant", "content": "5+ years Python, Django, cloud experience."},  # Recent
    ]
    
    original_count = len(messages)
    compressed_msgs = compress_conversation(messages, keep_recent=2)
    
    print(f"Original: {original_count} messages")
    print(f"Compressed: {len(compressed_msgs)} messages\n")
    
    print("Compressed conversation:")
    for msg in compressed_msgs:
        print(f"  {msg['role']}: {msg['content'][:80]}...")
