# Lesson 04: Conversations

## Overview
Building conversational AI requires managing message history, context windows, and providing good UX. These scripts cover the essentials.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_multi_turn.py` | Basic multi-turn conversation | `python 01_multi_turn.py` |
| `02_conversation_class.py` | Encapsulate conversation in a class | `python 02_conversation_class.py` |
| `03_context_window.py` | Truncate when context gets too large | `python 03_context_window.py` |
| `04_summarization.py` | Summarize old context instead of truncating | `python 04_summarization.py` |
| `05_interactive_chat.py` | Simple interactive chat loop | `python 05_interactive_chat.py` |
| `06_streaming_chat.py` | Interactive chat with streaming | `python 06_streaming_chat.py` |

## Key Takeaways

1. **You manage history** - OpenAI doesn't remember previous calls
2. **Message format**: `{"role": "user|assistant|system", "content": "..."}`
3. **Context window limits** - gpt-4o: 128k tokens, gpt-4o-mini: 128k tokens
4. **Truncation vs Summarization** - trade-off between simplicity and context preservation
5. **Streaming** - better UX for longer responses

## Conversation Architecture

```
User Input
    ↓
Add to message history
    ↓
[Context management: truncate/summarize if needed]
    ↓
Send to OpenAI
    ↓
Get response
    ↓
Add response to history
    ↓
Display to user
```

## For Your Job Chatbot

Your chatbot will need:
- System prompt defining the "job market analyst" persona
- Conversation history to maintain context
- Context management for long sessions
- Possibly: structured extraction for user intents

## Next Steps
→ Lesson 05: Embeddings (vector representations of text)
