# Module 3.1: Advanced Agent Patterns

> *"Sophisticated reasoning and self-improvement"*

This module covers advanced patterns that make agents more capable, reliable, and autonomous through better reasoning, planning, and self-correction.

## Scripts

| Script | Topic | Key Concept |
|--------|-------|-------------|
| `01_react_pattern.py` | ReAct Pattern | Reason → Act → Observe loop improves tool selection and error recovery |
| `02_reflection.py` | Self-Reflection | Agent critiques and improves its own output through self-evaluation |
| `03_planning.py` | Planning | Decompose complex tasks into ordered steps before execution |
| `04_self_correction.py` | Self-Correction | Detect validation errors and retry automatically |
| `05_confidence_routing.py` | Confidence-Based Routing | Route uncertain cases to stronger models based on confidence |
| `06_iterative_refinement.py` | Iterative Refinement | Generate → Evaluate → Refine loop improves quality |
| `07_tree_of_thought.py` | Tree of Thought | Explore multiple reasoning paths and select the best |

## Core Patterns

### 1. ReAct (Reasoning + Acting)
```
Input → THINK (reason about what to do)
      → ACT (use a tool)
      → OBSERVE (see result)
      → THINK (reason about next step)
      → ...
```
Better than acting blindly - explicit reasoning improves decisions.

### 2. Reflection
```
Generate → Critique → Refine → Critique → ...
```
Agent evaluates and improves its own work without human feedback.

### 3. Planning
```
Problem → Plan (ordered steps) → Execute steps → Result
```
Upfront planning reduces errors in complex multi-step tasks.

### 4. Self-Correction
```
Generate → Validate → If invalid: Fix and retry
```
Automatically recover from format errors, validation failures.

### 5. Confidence Routing
```
Fast Model → Low confidence? → Route to Stronger Model
```
Cost-efficient: use expensive models only when needed.

### 6. Iterative Refinement
```
v1 → Evaluate → v2 → Evaluate → v3 → ...
```
Each iteration builds on the last - converges to high quality.

### 7. Tree of Thought
```
Problem → [Path A, Path B, Path C] → Evaluate all → Pick best
```
More thorough than single-path - explores alternatives.

## Pattern Selection Guide

| Use Case | Best Pattern |
|----------|--------------|
| Tool-heavy tasks | ReAct |
| Quality-critical output | Reflection + Iterative Refinement |
| Complex multi-step problems | Planning |
| Structured data extraction | Self-Correction |
| Cost optimization | Confidence Routing |
| Open-ended reasoning | Tree of Thought |

## Job Data Application

These patterns excel at:
- **ReAct**: Job search with multiple filters and tools
- **Reflection**: Writing high-quality job descriptions
- **Planning**: Complex candidate matching workflows
- **Self-Correction**: Extracting structured job data
- **Confidence Routing**: Classifying job categories at scale
- **Iterative Refinement**: Creating polished job summaries
- **Tree of Thought**: Comparing multiple job offers

## Prerequisites

Install the required libraries:

```bash
pip install openai pydantic
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Running the Scripts

Each script is self-contained and can be run directly:

```bash
python 01_react_pattern.py
python 02_reflection.py
python 03_planning.py
# ... etc
```

## Key Insights

1. **Explicit reasoning improves results** - ReAct and planning outperform direct action
2. **Self-evaluation is powerful** - Agents can critique and improve their own work
3. **Multiple attempts > single attempt** - Refinement and self-correction beat one-shot
4. **Explore alternatives** - Tree of Thought finds better solutions than greedy search
5. **Route intelligently** - Confidence routing balances cost and accuracy

## Performance vs Cost Tradeoff

| Pattern | Latency | API Calls | Cost | Quality |
|---------|---------|-----------|------|---------|
| Direct (baseline) | 1x | 1 | $1 | ⭐⭐⭐ |
| ReAct | 2-5x | 3-10 | $3-10 | ⭐⭐⭐⭐ |
| Reflection | 2-3x | 2-4 | $2-4 | ⭐⭐⭐⭐ |
| Planning + Execution | 1.5-3x | 2-5 | $2-5 | ⭐⭐⭐⭐ |
| Self-Correction | 1-3x | 1-3 | $1-3 | ⭐⭐⭐⭐ |
| Confidence Routing | 1-2x | 1-2 | $1-5 | ⭐⭐⭐⭐ |
| Iterative Refinement | 3-5x | 4-8 | $4-8 | ⭐⭐⭐⭐⭐ |
| Tree of Thought | 5-10x | 6-15 | $6-15 | ⭐⭐⭐⭐⭐ |

## Book References

- `AI_eng.6` - Agent architecture, failure modes, and self-improvement
- `AI_eng.10.3` - Confidence scoring and routing strategies
- `hands_on_LLM.II.6` - Tree of Thought and advanced reasoning
- `hands_on_LLM.II.7` - ReAct pattern and agent loops
- `NLP_cook.10` - Conversational agents and planning

## Next Steps

After mastering these patterns:
- Module 3.2: Multi-Agent Systems
- Module 3.3: Advanced Memory Patterns
- Module 3.4: Advanced RAG Techniques
