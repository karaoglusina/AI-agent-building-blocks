# Conversations

A single API call answers a question. A conversation answers follow-up questions, remembers context, and feels like talking to someone who pays attention.

## Why This Matters

Users don't interact with AI in single shots. They ask a question, then refine it. They say "show me more like this" or "actually, I meant remote jobs." Without conversation management, each message starts from scratch.

For our job market analyzer, conversations let users naturally explore: "Show me ML engineer roles" → "Which ones are remote?" → "Compare the top two" → "Tell me more about the second one."

## The Key Ideas

### You Manage History

This is the key insight: OpenAI doesn't remember previous calls. You send the entire conversation each time:

```python
messages = [
    {"role": "system", "content": "You are a job market analyst."},
    {"role": "user", "content": "Show me ML engineer roles"},
    {"role": "assistant", "content": "Here are the top ML engineer roles..."},
    {"role": "user", "content": "Which ones are remote?"}  # New message
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages  # Send everything
)
```

The model sees the full context and can reference earlier messages.

### Context Windows Have Limits

Models have a maximum context length:
- gpt-4o: 128k tokens
- gpt-4o-mini: 128k tokens

That's roughly 100k words - plenty for most conversations. But if you're including retrieved documents, tool outputs, and long chat histories, you can hit limits.

### Truncation: The Simple Approach

When context gets too long, keep only the most recent messages:

```python
def truncate_messages(messages, max_messages=20):
    if len(messages) <= max_messages:
        return messages
    # Keep system prompt + recent messages
    return [messages[0]] + messages[-max_messages:]
```

Simple, but you lose early context. The user says "like you mentioned earlier" and the model has no idea what they mean.

### Summarization: Preserve Context

Better approach: summarize old messages instead of deleting them:

```python
# When conversation gets long:
# 1. Take old messages
# 2. Ask model to summarize them
# 3. Replace old messages with summary
# 4. Continue conversation

summary = summarize_messages(old_messages)
messages = [
    system_prompt,
    {"role": "system", "content": f"Earlier conversation summary: {summary}"},
    *recent_messages
]
```

More complex, but preserves important context.

### Streaming for UX

Long responses should stream. Users see progress instead of waiting:

```python
for chunk in client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True
):
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_multi_turn.py | Basic multi-turn conversation |
| 02_conversation_class.py | Encapsulate conversation logic in a class |
| 03_context_window.py | Truncate when context gets too large |
| 04_summarization.py | Summarize old context instead of truncating |
| 05_interactive_chat.py | Simple interactive chat loop |
| 06_streaming_chat.py | Interactive chat with streaming responses |

## Things to Think About

- **When should you truncate vs summarize?** Truncation is simpler and cheaper. Summarization preserves more context but adds latency and cost. For casual chat, truncate. For complex multi-step tasks, summarize.
- **What belongs in the system prompt vs conversation history?** System prompt: persona, rules, capabilities. History: what actually happened in this session.
- **How do you handle errors mid-conversation?** The model might refuse a request or fail. Your conversation class needs to handle this gracefully.

## Related

- [Memory Patterns](../phase-2-building-ai-systems/memory-patterns.md) - Sophisticated approaches to remembering
- [Context Engineering](../phase-2-building-ai-systems/context-engineering.md) - Deciding what to include

## Book References

- hands_on_LLM.II.7 - Conversation patterns
- speach_lang.II.15 - Dialogue systems
