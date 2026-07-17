# Lesson 03: Structured Output

## Overview
This is where OpenAI and Pydantic come together. Structured output ensures LLM responses are valid, typed, and predictable.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_json_mode.py` | Force JSON output | `python 01_json_mode.py` |
| `02_structured_output.py` | Pydantic schema for output | `python 02_structured_output.py` |
| `03_extraction.py` | Extract structured data from text | `python 03_extraction.py` |
| `04_classification.py` | Classify into predefined categories | `python 04_classification.py` |
| `05_complex_extraction.py` | Nested models for rich extraction | `python 05_complex_extraction.py` |
| `06_batch_processing.py` | Process multiple items | `python 06_batch_processing.py` |
| `07_schema_inspection.py` | See the JSON schema Pydantic generates | `python 07_schema_inspection.py` |

## Key Takeaways

1. **`response_format={"type": "json_object"}`** forces the model to emit valid JSON
2. **JSON mode guarantees syntax, not schema** — the model can still return the wrong fields
3. **`YourModel.model_validate_json(...)`** is what actually enforces the schema, and gives you a typed Pydantic object
4. **Literal types** restrict values to specific options (great for classification)
5. **Nested models** enable complex, hierarchical extraction
6. **Field descriptions** help the model understand what to extract

## The Magic

```python
# Define what you want
class Result(BaseModel):
    name: str
    score: int

# Ask for JSON
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "... return JSON with name and score"}],
    response_format={"type": "json_object"},  # guarantees valid JSON, not this schema
)

# Pydantic is what enforces the shape
result: Result = Result.model_validate_json(response.choices[0].message.content)
```

## Why This Matters for Agents

- **Tool outputs** can be structured (agent knows what it got back)
- **Decision making** can use classification (route to right handler)
- **Information extraction** enables working with unstructured text
- **No parsing errors** - the schema is enforced by OpenAI

## Next Steps
→ Lesson 04: Conversations (multi-turn, context management)
