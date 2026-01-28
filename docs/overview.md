# The Landscape

Before diving into code, it helps to have a mental map of what's out there. When I started learning AI engineering, I kept encountering terms and patterns without understanding how they fit together. This overview is what I wish I had read first.

## The Basic Building Blocks

At the core, you're working with a few primitives:

### LLMs (Large Language Models)
The models themselves. You send text in, you get text out. They're surprisingly capable at understanding context, following instructions, and generating coherent responses. They're also surprisingly bad at math, precise recall, and knowing when they're wrong.

**Key insight**: LLMs are completion engines. They predict what comes next. Everything else - chatbots, agents, RAG - is built on top of this basic capability.

### Embeddings
Vector representations of text. "Machine learning engineer" and "ML engineer" have different words but nearly identical embeddings because they mean the same thing. This is how you do semantic search instead of keyword matching.

**Key insight**: Similar meanings → similar vectors → you can find related content by comparing vectors.

### Vector Databases
Databases optimized for storing and searching embeddings. ChromaDB, Pinecone, pgvector, Weaviate - they all do roughly the same thing: let you store millions of vectors and quickly find the most similar ones to a query.

**Key insight**: Vector search is the foundation of retrieval. Without it, you're stuck with keyword matching or stuffing everything into the prompt.

## The Common Patterns

Once you understand the building blocks, patterns emerge:

### RAG (Retrieval-Augmented Generation)
Instead of hoping the LLM knows the answer, you first retrieve relevant documents and include them in the prompt. "Here are the relevant job postings. Now answer the user's question."

This is probably the most useful pattern. It grounds the LLM in your data, reduces hallucination, and lets you work with information that wasn't in the training data.

### Agents
LLMs in a loop with tools. The model decides what to do, calls a tool, observes the result, decides what to do next. Keep going until the task is done.

Agents can search databases, call APIs, run calculations - anything you can wrap in a function. The LLM acts as the orchestrator, deciding which tool to use when.

### Memory
How do you give a conversation context? How do you remember that the user prefers remote jobs? Memory patterns handle this - from simple conversation history to sophisticated retrieval of past interactions.

### Chains and Pipelines
Breaking complex tasks into steps. Extract information → classify it → route to handler → generate response. Each step is simple; the composition handles complexity.

## What Makes This Hard

AI engineering isn't hard because the APIs are complex. It's hard because:

**Outputs are non-deterministic.** The same prompt can give different results. You can't write tests the way you test normal code.

**Failure modes are subtle.** The model doesn't crash - it just gives a confident wrong answer. Detecting when the model is wrong is often harder than getting the right answer.

**Context is limited.** You can't just feed everything to the model. A 10,000 job post dataset is ~50MB of text. You need strategies for what to include and what to leave out.

**Latency and cost add up.** Each API call takes time and money. An agent that makes 10 LLM calls to answer a question might be impressive, but it's also slow and expensive.

**Evaluation is tricky.** How do you know if your RAG system is good? If your agent is reliable? Traditional metrics only get you so far.

## The Technology Stack

Here's what a typical production AI system looks like:

```
┌─────────────────────────────────────────────────┐
│                   Application                    │
│    (FastAPI, Flask, or whatever framework)       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Orchestration Layer                 │
│  (Your code: chains, agents, routing, memory)    │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │   LLM    │  │  Vector  │  │ Database │
   │   API    │  │    DB    │  │   (SQL)  │
   └──────────┘  └──────────┘  └──────────┘
```

The orchestration layer is where you live as an AI engineer. You're not training models - you're composing capabilities into systems that solve problems.

## Where to Start

If you're building something like a job market analyzer, here's roughly the order things get useful:

1. **Call the LLM** (Phase 1): Just get something working
2. **Structure outputs** (Phase 1): Make the LLM return data you can parse
3. **Embed and search** (Phase 1): Find relevant content semantically
4. **Build RAG** (Phase 2): Answer questions grounded in your data
5. **Add tools and agents** (Phase 2): Let the system take actions
6. **Add memory** (Phase 2): Remember across conversations
7. **Evaluate** (Phase 2): Know when things are working

Everything after that is refinement - better RAG, more sophisticated agents, production hardening.

## The 80/20 Rule

Phase 2 of this curriculum covers ~80% of what you'll use day-to-day. RAG, agents, memory, prompting - these patterns appear everywhere.

Phases 3-5 add nuance: advanced agent patterns, multi-agent systems, production deployment, fine-tuning. They matter when you need them, but start with the fundamentals.

## A Word on Frameworks

LangChain, LlamaIndex, and similar frameworks abstract away the mechanics. That's good for prototyping but problematic for understanding. When something breaks, you need to know what's happening underneath.

This curriculum deliberately avoids frameworks. We write the orchestration ourselves. It's more code, but you understand every line.

Once you understand the patterns, frameworks become tools you can use (or not) as appropriate. But starting with frameworks is like learning to drive in a self-driving car - you'll be lost when you need to take manual control.

## Next Steps

Ready to start? Head to [Phase 1: Foundations](./phase-1-foundations/0.0-phase-1-foundations-index.md) to learn the basics.
