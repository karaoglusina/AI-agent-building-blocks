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

| Phase | What It Covers |
|-------|----------------|
| [Phase 1: Foundations](./phase-1-foundations/index.md) | OpenAI API, Pydantic, embeddings, vector search |
| [Phase 2: Building AI Systems](./phase-2-building-ai-systems/index.md) | Text prep, RAG, agents, memory, prompting, evaluation |
| [Phase 3: Advanced Patterns](./phase-3-advanced-patterns/index.md) | ReAct, multi-agent, advanced RAG, FastAPI, clustering |
| [Phase 4: Production](./phase-4-production/index.md) | Docker, PostgreSQL, observability, guardrails, deployment |
| [Phase 5: Specialization](./phase-5-specialization/index.md) | Fine-tuning, custom embeddings, multimodal |

## The Philosophy

A few principles guided how I put this together:

**No frameworks.** LangChain and LlamaIndex are great for getting started, but they hide the mechanics. Here, we write the orchestration ourselves so we understand what's actually happening.

**One concept per file.** Each script teaches one thing. If you want to learn about tool calling, there's a single file for that. No hunting through a 500-line application.

**Real data.** The job posts dataset is messy and real. You'll encounter encoding issues, missing fields, and the kinds of problems you face in production.

**Practical over theoretical.** No deep dives into attention mechanisms. Just "here's how you use this" with working code.

## Running the Scripts

Most scripts can be run directly:

```bash
python modules/phase1/1.1-openai-basics/01_basic_call.py
```

You'll need:
- Python 3.12+
- An OpenAI API key (`export OPENAI_API_KEY="sk-..."`)
- The dependencies for each phase (see the module READMEs)

## What's Next

Start with the [overview](./overview.md) to get a sense of the landscape, then dive into Phase 1 when you're ready.
