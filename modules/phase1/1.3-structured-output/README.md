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

1. **`responses.parse()`** instead of `responses.create()` for structured output
2. **`text_format=YourModel`** tells OpenAI the expected schema
3. **`response.output_parsed`** is already a Pydantic model
4. **Literal types** restrict values to specific options (great for classification)
5. **Nested models** enable complex, hierarchical extraction
6. **Field descriptions** help the model understand what to extract

## The Magic

```python
# Define what you want
class Result(BaseModel):
    name: str
    score: int

# Ask for it
response = client.responses.parse(
    model="gpt-4o-mini",
    input="...",
    text_format=Result  # OpenAI guarantees this schema
)

# Get exactly what you defined
result: Result = response.output_parsed
```

## Why This Matters for Agents

- **Tool outputs** can be structured (agent knows what it got back)
- **Decision making** can use classification (route to right handler)
- **Information extraction** enables working with unstructured text
- **No parsing errors** - the schema is enforced by OpenAI

## Next Steps
→ Lesson 04: Conversations (multi-turn, context management)
