# Phase 1: Foundations

This is where it all starts. Before building RAG systems or agents, you need to understand the basic pieces: how to call an LLM, how to structure its outputs, how to represent text as vectors.

## What You're Building Toward

By the end of Phase 1, you'll be able to:
- Call OpenAI's API and handle responses properly
- Validate and structure data with Pydantic
- Make the LLM return structured JSON you can parse
- Maintain conversation history
- Convert text to embeddings and search by meaning
- Use a vector database for semantic search

For our job market analyzer, this means you can embed job descriptions, search for similar roles, and get the LLM to extract structured information from postings.

## The Modules

### <a href="./1.1-openai-basics.md">OpenAI Basics</a>
The starting point. How to make API calls, handle responses, use different models, stream output. If you've never called an LLM API, start here.

### <a href="./1.2-pydantic-basics.md">Pydantic Basics</a>
Data validation and type safety in Python. This becomes essential when you're parsing LLM outputs and working with structured data. Not AI-specific, but foundational for everything that follows.

### <a href="./1.3-structured-output.md">Structured Output</a>
Getting the LLM to return JSON that matches a schema. This is the bridge between natural language and data you can actually use in code. Extraction, classification, complex responses - it all runs through structured output.

### <a href="./1.4-conversations.md">Conversations</a>
Moving beyond single prompts. How to maintain context across messages, handle long conversations, build interactive chat applications. The foundation for any conversational AI.

### <a href="./1.5-embeddings.md">Embeddings</a>
Text as vectors. This is how you find similar content, do semantic search, and build RAG systems. Probably the most important concept in the whole curriculum.

### <a href="./1.6-vector-search.md">Vector Search</a>
Using ChromaDB to store and search embeddings at scale. When you have 10,000 job postings, you need a real database - not just numpy arrays.

## The Flow

These modules build on each other:

```
OpenAI Basics → Conversations
      ↓
Pydantic → Structured Output
      ↓
Embeddings → Vector Search
```

Start with OpenAI Basics if you're new. If you already know how to call the API, you could jump to Structured Output or Embeddings depending on what you need.

## What You'll Need

```bash
pip install openai pydantic chromadb numpy tiktoken
```

And set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Time Investment

If you're new to this, expect to spend meaningful time here. These concepts show up everywhere - rushing through them means struggling later. If you already work with LLMs, you might skim through quickly and focus on the bits that are new to you.
