# Lesson 02: Pydantic Basics

## Overview
Pydantic is the foundation for structured data in Python AI applications. It validates data, converts types, and generates JSON schemas that OpenAI uses for structured output.

## Scripts

| File | Concept | Run it |
|------|---------|--------|
| `01_basic_model.py` | Define models with type hints | `python 01_basic_model.py` |
| `02_field_types.py` | Common types: str, int, list, dict, datetime | `python 02_field_types.py` |
| `03_validation.py` | Field constraints and custom validators | `python 03_validation.py` |
| `04_nested_models.py` | Models containing other models | `python 04_nested_models.py` |
| `05_json_serialization.py` | Convert to/from JSON and dicts | `python 05_json_serialization.py` |
| `06_from_raw_data.py` | Parse your job data into models | `python 06_from_raw_data.py` |
| `07_model_methods.py` | Add computed fields and methods | `python 07_model_methods.py` |
| `08_enums_literals.py` | Restrict values to specific options | `python 08_enums_literals.py` |

## Key Takeaways

1. **BaseModel** is the foundation - inherit from it
2. **Type hints** are enforced, not just documentation
3. **model_dump()** → dict, **model_dump_json()** → JSON string
4. **model_validate()** → create from dict, **model_validate_json()** → from string
5. **Field()** adds constraints (min/max, regex, etc.)
6. **Nested models** compose complex structures
7. **extra = "ignore"** skips fields not in your model

## Why Pydantic Matters for AI

- OpenAI's **structured output** uses Pydantic schemas
- Ensures LLM outputs are **valid and typed**
- Makes data processing **reliable and predictable**

## Next Steps
→ Lesson 03: Structured Output (combining OpenAI + Pydantic)
