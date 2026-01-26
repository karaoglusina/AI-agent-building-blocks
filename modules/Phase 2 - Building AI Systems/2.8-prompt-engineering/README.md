# Module 2.8: Prompt Engineering

> *"Maximize LLM performance through better prompts"*

This module covers techniques for crafting effective prompts that produce reliable, high-quality outputs.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_system_prompt_design.py` | System Prompt Design | Good prompts define role, capabilities, constraints, format |
| `02_chain_of_thought.py` | Chain of Thought | Explicit reasoning steps improve accuracy |
| `03_few_shot_examples.py` | Few-Shot Examples | Examples show exactly what you want |
| `04_output_formatting.py` | Output Formatting | Clear instructions produce consistent output |
| `05_instructor_basics.py` | Instructor Library | Model-agnostic structured output |
| `06_constrained_generation.py` | Constrained Generation | Constraints ensure output matches expected formats |
| `07_defensive_prompting.py` | Defensive Prompting | Anticipate misuse and build defenses |
| `08_self_consistency.py` | Self-Consistency | Multiple samples with voting improves reliability |

## Prompt Engineering Principles

1. **Be specific**: Vague prompts get vague answers
2. **Show examples**: Few-shot > zero-shot for consistency
3. **Structure output**: Format instructions prevent parsing issues
4. **Think step-by-step**: CoT for complex reasoning
5. **Defend**: Never trust user input completely

## Prerequisites

Install the required libraries:

```bash
pip install openai instructor pydantic
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_system_prompt_design.py
python 02_chain_of_thought.py
# ... etc
```

## Technique Selection Guide

| Technique | When to Use | Cost Impact |
|-----------|-------------|-------------|
| System Prompt | Always | None |
| Chain of Thought | Complex reasoning | +20-50% tokens |
| Few-Shot | Need consistent format | +50-100% tokens |
| Constrained | Need validated output | None |
| Self-Consistency | High-stakes decisions | N× calls |

## Book References

- `AI_eng.5` - Prompt engineering fundamentals
- `AI_eng.2` - Structured output (Instructor)
- `hands_on_LLM.II.6` - Advanced prompting techniques
- `speach_lang.I.12.4` - Chain of thought reasoning
