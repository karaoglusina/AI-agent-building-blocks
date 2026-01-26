# Advanced Agent Patterns

Basic agents call tools in a loop. Advanced agents reason about what to do, reflect on their outputs, plan multi-step tasks, and correct their own mistakes.

## Why This Matters

Simple agents work for simple tasks. But complex tasks - like "research the AI job market in Europe and create a comparison report" - need more sophisticated reasoning. Advanced patterns make agents more capable and reliable.

For our job market analyzer, these patterns enable genuine research tasks: searching, comparing, synthesizing, and refining outputs into something actually useful.

## The Key Ideas

### ReAct: Reason + Act

Instead of blindly calling tools, reason first:

```
Input → THINK: "I need to find remote ML jobs first"
      → ACT: search_jobs("remote ML engineer")
      → OBSERVE: [5 results]
      → THINK: "Now I should compare salaries"
      → ACT: analyze_salaries(results)
      → ...
```

Explicit reasoning improves tool selection and error recovery. The model explains what it's doing and why.

### Self-Reflection

After generating output, critique it:

```
Generate response → "Here are the top 3 ML jobs..."
Critique → "This doesn't address the salary question"
Refine → "Here are the top 3 ML jobs with salary info..."
```

Reflection catches errors and improves quality without human intervention.

### Planning

For complex tasks, plan before executing:

```
Task: "Create a report comparing job markets in 3 cities"

Plan:
1. Search jobs in City A
2. Search jobs in City B
3. Search jobs in City C
4. Extract key metrics from each
5. Compare and synthesize
6. Format report

Then execute each step.
```

Upfront planning reduces errors in multi-step tasks.

### Self-Correction

When something fails, fix it automatically:

```
Generate structured output → Validation fails
Detect error: "salary_min should be integer, got string"
Retry with correction prompt → Valid output
```

Build in retry logic with specific error feedback.

### Confidence Routing

Not every query needs the most powerful model:

```
Query → Fast model (gpt-4o-mini) → Check confidence
      → High confidence? Return result
      → Low confidence? Route to stronger model (gpt-4o)
```

Cost-efficient: use expensive models only when needed.

### Iterative Refinement

Generate, evaluate, improve:

```
v1: Initial draft
Evaluate: 7/10 - needs more specific examples
v2: Refined with examples
Evaluate: 8/10 - still missing salary data
v3: Added salary comparisons
Evaluate: 9/10 - good enough
```

Each iteration builds on the last.

### Tree of Thought

Explore multiple reasoning paths:

```
Problem: "Which job is best for career growth?"

Path A: Prioritize salary growth
Path B: Prioritize skill development
Path C: Prioritize company reputation

Evaluate all paths → Select best reasoning
```

More thorough than single-path reasoning.

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_react_pattern.py | Reason-Act-Observe loop |
| 02_reflection.py | Self-critique and improvement |
| 03_planning.py | Decompose then execute |
| 04_self_correction.py | Automatic error recovery |
| 05_confidence_routing.py | Route by model confidence |
| 06_iterative_refinement.py | Multi-pass improvement |
| 07_tree_of_thought.py | Explore multiple paths |

## Pattern Selection

| Use Case | Best Pattern |
|----------|--------------|
| Tool-heavy tasks | ReAct |
| Quality-critical output | Reflection + Refinement |
| Complex multi-step problems | Planning |
| Structured extraction | Self-Correction |
| Cost optimization | Confidence Routing |
| Open-ended reasoning | Tree of Thought |

## Cost vs Quality

| Pattern | API Calls | Quality |
|---------|-----------|---------|
| Direct (baseline) | 1 | ⭐⭐⭐ |
| ReAct | 3-10 | ⭐⭐⭐⭐ |
| Reflection | 2-4 | ⭐⭐⭐⭐ |
| Planning | 2-5 | ⭐⭐⭐⭐ |
| Iterative Refinement | 4-8 | ⭐⭐⭐⭐⭐ |
| Tree of Thought | 6-15 | ⭐⭐⭐⭐⭐ |

More capable patterns cost more. Choose based on task requirements.

## Things to Think About

- **When is the extra cost worth it?** For high-value outputs where quality matters. Not for every routine query.
- **How do you know when to stop iterating?** Set quality thresholds or max iterations. Diminishing returns after a point.
- **Can patterns be combined?** Yes - Planning + ReAct + Reflection for maximum capability.

## Related

- [Agent Orchestration](../phase-2-building-ai-systems/agent-orchestration.md) - Basic agent patterns
- [Multi-Agent Systems](./multi-agent-systems.md) - Coordinating multiple agents
- [Evaluation Systems](./evaluation-systems.md) - Measuring agent quality

## Book References

- AI_eng.6 - Agent architecture and failure modes
- hands_on_LLM.II.6 - Tree of Thought
- hands_on_LLM.II.7 - ReAct pattern
