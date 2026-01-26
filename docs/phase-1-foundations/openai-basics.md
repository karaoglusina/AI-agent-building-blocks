# OpenAI Basics

Everything starts here. Before RAG, before agents, before embeddings - you need to know how to make an API call and handle the response.

## Why This Matters

If you're building an AI system, you'll be making hundreds or thousands of API calls. Understanding the basics - how to structure requests, what the response looks like, when to stream - saves you from hitting walls later.

For our job market analyzer, this is how we ask questions about job postings, extract information, and generate summaries.

## The Key Ideas

### The API Call

At its simplest:
```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)
```

That's it. Everything else is refinement.

### System Prompts

The system prompt shapes behavior. It's how you tell the model who it is and how to act:

```python
messages = [
    {"role": "system", "content": "You are a job market analyst who gives concise insights."},
    {"role": "user", "content": "What skills are most in demand?"}
]
```

The system prompt is your main lever for controlling output quality and style.

### Temperature

Temperature controls randomness:
- `temperature=0`: Deterministic, focused. Use for extraction, classification.
- `temperature=0.7`: Balanced creativity. Good default.
- `temperature=1.5+`: More random, creative. Rarely what you want.

### Streaming

For longer responses, streaming improves UX. Instead of waiting for the complete response, you get tokens as they're generated:

```python
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

Users see progress instead of staring at a loading spinner.

### Models

- **gpt-4o-mini**: Cheap and fast. Use for learning and most tasks.
- **gpt-4o**: More capable, more expensive. Use when quality matters.
- **gpt-4-turbo**: Older model, being phased out.

Start with gpt-4o-mini. You can always upgrade later.

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_basic_call.py | Simplest possible API call |
| 02_input_formats.py | String vs message list input |
| 03_system_prompt.py | Using system prompts to control behavior |
| 04_parameters.py | Temperature, max_tokens, and other parameters |
| 05_response_object.py | Understanding the response structure |
| 06_streaming.py | Stream tokens as they arrive |
| 07_models.py | Different models and when to use them |
| 08_error_handling.py | Handling API errors gracefully |

## Things to Think About

- **What happens when the API fails?** Rate limits, timeouts, network errors - you need to handle these.
- **Why not just pass everything as a single string?** Message format separates roles, making it clear what's system context, what's user input, what's previous conversation.
- **When does temperature=0 still give different outputs?** Even with temperature=0, there's some non-determinism. Don't rely on exact reproducibility.

## Related

- [Structured Output](./structured-output.md) - Making the LLM return parseable data
- [Conversations](./conversations.md) - Managing multi-turn interactions
