# AI Engineering Notes

This isn't a formal course. It's a collection of notes I put together while learning to build AI systems - specifically, how to take the building blocks (LLMs, embeddings, vector databases) and assemble them into something useful.

## What's Here

About 170 Python scripts organized into phases, each demonstrating one concept. The scripts are short (40-80 lines), self-contained, and runnable. No magic, no opaque frameworks - just Python code you can read and understand.

## The Unifying Example

Throughout these notes, I keep coming back to one scenario: **building an agent that analyzes job postings to understand the AI job market.**

Why this example? Because it touches on almost everything:

- **Embeddings**: Find jobs similar to one you like
- **RAG**: Answer questions about the market without feeding 10k posts to the LLM
- **Classification**: Categorize roles (ML engineer vs researcher vs data scientist)
- **Agents**: Autonomously research, compare, and synthesize insights
- **Memory**: Remember what the user is looking for

The dataset (`data/job_post_data.json`) contains ~10,000 real job postings. It's messy, diverse, and large enough to hit real-world issues.

## How to Use This

**If you're new to AI engineering:**

Start with Phase 1. It covers the basics - calling OpenAI, validating data with Pydantic, understanding embeddings. Each script is designed to teach one thing.

**If you have some experience:**

Jump to whatever interests you. The modules are mostly independent. Phase 2 covers the patterns you'll use daily (RAG, agents, memory). Phase 3 and beyond goes deeper.

**If you're building something:**

Use this as a reference. When you need to implement RAG with citations, there's a script for that. When you need to handle agent errors gracefully, there's a script for that too.

## The Phases

| Phase                                                                                                | What It Covers                                            |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| <a href="./phase-1-foundations/0.0-phase-1-foundations-index.md">Phase 1: Foundations</a>                       | OpenAI API, Pydantic, embeddings, vector search           |
| <a href="./phase-2-building-ai-systems/0.0-phase-2-building-ai-systems-index.md">Phase 2: Building AI Systems</a> | Text prep, RAG, agents, memory, prompting, evaluation     |
| <a href="./phase-3-advanced-patterns/0.0-phase-3-advanced-patterns-index.md">Phase 3: Advanced Patterns</a>     | ReAct, multi-agent, advanced RAG, FastAPI, clustering     |
| <a href="./phase-4-production/0.0-phase-4-production-index.md">Phase 4: Production</a>                          | Docker, PostgreSQL, observability, guardrails, deployment |
| <a href="./phase-5-specialization/0.0-phase-5-specialization-index.md">Phase 5: Specialization</a>              | Fine-tuning, custom embeddings, multimodal                |

## Running the Scripts

Most scripts can be run directly. 

```bash
python modules/phase1/1.1-openai-basics/01_basic_call.py
```

You'll need:

- Python 3.12+
- An OpenAI API key (`export OPENAI_API_KEY="sk-..."`)
- The dependencies for each phase (see the module READMEs)

## What's Next

Start with the <a href="./overview.md">overview</a> to get a sense of the landscape, then dive into Phase 1 when you're ready.