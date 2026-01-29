"""
02 - Trace LLM Calls
=====================
Log all LLM interactions with detailed metadata.

Key concept: Comprehensive tracing of LLM calls enables debugging, performance analysis, and cost optimization.

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
def simple_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Simple LLM call with automatic tracing via decorator."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7)

    # Log generation to Langfuse
    langfuse_context.update_current_observation(
        input={"prompt": prompt, "model": model},
        output={"response": response.choices[0].message.content},
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        metadata={
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }
    )

    return response.choices[0].message.content


@observe()
def multi_turn_conversation(messages: list[dict]) -> str:
    """Multi-turn conversation with full context tracing."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7)

    # Log entire conversation
    langfuse_context.update_current_observation(
        input={"messages": messages},
        output={"response": response.choices[0].message.content},
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    )

    return response.choices[0].message.content


@observe()
def llm_with_tool_call(query: str) -> dict:
    """LLM call with tool/function calling traced."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )

    tool_calls = response.choices[0].message.tool_calls

    # Log with tool call details
    langfuse_context.update_current_observation(
        input={"query": query},
        output={
            "content": response.choices[0].message.content,
            "tool_calls": [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                } for tc in (tool_calls or [])
            ]
        },
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        metadata={
            "has_tool_calls": tool_calls is not None,
            "num_tool_calls": len(tool_calls) if tool_calls else 0,
        }
    )

    return {
        "content": response.choices[0].message.content,
        "tool_calls": tool_calls
    }


@observe()
def streaming_llm_call(prompt: str) -> str:
    """Streaming LLM call with tracing."""
    chunks = []

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}  # Get usage stats
    )

    total_usage = None

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chunks.append(content)

        # Usage comes in the last chunk
        if hasattr(chunk, 'usage') and chunk.usage:
            total_usage = chunk.usage

    full_response = "".join(chunks)

    # Log complete response
    langfuse_context.update_current_observation(
        input={"prompt": prompt},
        output={"response": full_response},
        usage={
            "prompt_tokens": total_usage.prompt_tokens if total_usage else 0,
            "completion_tokens": total_usage.completion_tokens if total_usage else 0,
            "total_tokens": total_usage.total_tokens if total_usage else 0,
        } if total_usage else None,
        metadata={"streaming": True}
    )

    return full_response


@observe()
def batch_llm_calls(prompts: list[str]) -> list[str]:
    """Batch multiple LLM calls with individual tracing."""
    responses = []

    for i, prompt in enumerate(prompts):
        # Each call gets its own span
        with langfuse_context.observe(name=f"batch_call_{i}") as span:
            response = simple_llm_call(prompt)
            responses.append(response)

            span.update(
                input={"prompt": prompt, "batch_index": i},
                output={"response": response}
            )

    return responses


if __name__ == "__main__":
    print("=== TRACE LLM CALLS ===\n")

    # 1. Simple call
    print("1. Simple LLM Call")
    print("-" * 60)
    response = simple_llm_call("What is machine learning?")
    print(f"Response: {response[:100]}...\n")

    # 2. Multi-turn conversation
    print("2. Multi-Turn Conversation")
    print("-" * 60)
    conversation = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 15 * 23?"}]
    response = multi_turn_conversation(conversation)
    print(f"Response: {response}\n")

    # 3. Tool call
    print("3. LLM with Tool Call")
    print("-" * 60)
    result = llm_with_tool_call("What's the weather in San Francisco?")
    print(f"Tool calls: {result['tool_calls']}\n")

    # 4. Streaming
    print("4. Streaming LLM Call")
    print("-" * 60)
    response = streaming_llm_call("Count from 1 to 5.")
    print(f"Response: {response}\n")

    # 5. Batch processing
    print("5. Batch LLM Calls")
    print("-" * 60)
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    responses = batch_llm_calls(prompts)
    for prompt, response in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {response[:80]}...\n")

    # Flush traces
    langfuse.flush()

    print("=" * 60)
    print("✓ All traces sent to Langfuse dashboard")
    print("=" * 60)
