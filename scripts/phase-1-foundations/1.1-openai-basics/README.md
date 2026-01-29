# Lesson 01: OpenAI API Basics

## Overview
These scripts introduce the fundamental concepts of the OpenAI Responses API.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_basic_call.py` | Simplest API call | `python 01_basic_call.py` |
| `02_input_formats.py` | String vs message list input | `python 02_input_formats.py` |
| `03_system_prompt.py` | System prompts for behavior control | `python 03_system_prompt.py` |
| `04_parameters.py` | temperature, max_tokens | `python 04_parameters.py` |
| `05_response_object.py` | Understanding the response structure | `python 05_response_object.py` |
| `06_streaming.py` | Stream tokens as they arrive | `python 06_streaming.py` |
| `07_models.py` | Different models and when to use them | `python 07_models.py` |
| `08_error_handling.py` | Handling API errors | `python 08_error_handling.py` |

## Key Takeaways

1. **`client.responses.create()`** is the main method
2. **Input** can be a string or list of messages
3. **System prompts** shape AI behavior
4. **Temperature** controls creativity (0=focused, 2=random)
5. **Streaming** is useful for UX (shows progress)
6. **gpt-4o-mini** is best for learning (cheap + fast)

## Next Steps
→ Lesson 02: Pydantic Basics
