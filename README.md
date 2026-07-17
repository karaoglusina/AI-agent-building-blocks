# AI Agent Building Blocks

[![Scripts](https://github.com/karaoglusina/AI-agent-building-blocks/actions/workflows/scripts.yml/badge.svg)](https://github.com/karaoglusina/AI-agent-building-blocks/actions/workflows/scripts.yml)

A Python reference for learning and building AI agents from scratch — 208 short scripts covering the stack from API basics to production patterns.

> **Status:** these are my study notes from moving out of academic AI research into AI engineering, published because they might be useful to someone on the same path. They're maintained on a best-effort basis, not actively developed. Corrections and issues are welcome; I may be slow to respond.

## Why This Exists

Building AI agents requires understanding dozens of interconnected components. Without a map of the landscape, you learn reactively — discovering critical patterns only after committing to an architecture.

This repo is a **map of the AI agent landscape** — broad rather than deep, by design. Build a mental repertoire upfront so when you design real solutions, you'll recognize which patterns apply and what's in your toolchain. See [Scope & Limitations](#scope--limitations) for where the map stops.

Each script isolates one specific concept so you can:

- See the full range of components before committing to an architecture
- Understand the underlying patterns that persist across frameworks and libraries
- Reference working examples when composing solutions
- Make informed design decisions instead of discovering missing pieces mid-build

## What's Covered

| Phase                                                                                                | Focus                                          | Modules (docs)                                                                                       |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **[Phase 1: Foundations](modules/phase-1-foundations/0.0-phase-1-foundations-index.md)**             | OpenAI API + core building blocks              | [OpenAI API](modules/phase-1-foundations/1.1-openai-basics.md) · [Pydantic Models](modules/phase-1-foundations/1.2-pydantic-basics.md) · [Structured Output](modules/phase-1-foundations/1.3-structured-output.md) · [Conversations](modules/phase-1-foundations/1.4-conversations.md) · [Embeddings](modules/phase-1-foundations/1.5-embeddings.md) · [Vector Search](modules/phase-1-foundations/1.6-vector-search.md) |
| **[Phase 2: Core AI Engineering](modules/phase-2-building-ai-systems/0.0-phase-2-building-ai-systems-index.md)** | Daily patterns for building AI systems         | [Text Preparation](modules/phase-2-building-ai-systems/2.1-text-preparation.md) · [Information Extraction](modules/phase-2-building-ai-systems/2.2-information-extraction.md) · [Classification & Routing](modules/phase-2-building-ai-systems/2.3-classification-routing.md) · [RAG Pipeline](modules/phase-2-building-ai-systems/2.4-rag-pipeline.md) · [Agent Orchestration](modules/phase-2-building-ai-systems/2.5-agent-orchestration.md) · [Context Engineering](modules/phase-2-building-ai-systems/2.6-context-engineering.md) · [Memory Patterns](modules/phase-2-building-ai-systems/2.7-memory-patterns.md) · [Prompt Engineering](modules/phase-2-building-ai-systems/2.8-prompt-engineering.md) · [Evaluation](modules/phase-2-building-ai-systems/2.9-evaluation-basics.md) |
| **[Phase 3: Advanced Patterns](modules/phase-3-advanced-patterns/0.0-phase-3-advanced-patterns-index.md)** | Sophisticated agent behaviors + deeper systems | [Advanced Agents](modules/phase-3-advanced-patterns/3.1-advanced-agent-patterns.md) · [Multi-Agent Systems](modules/phase-3-advanced-patterns/3.2-multi-agent-systems.md) · [Advanced Memory](modules/phase-3-advanced-patterns/3.3-advanced-memory.md) · [Advanced RAG](modules/phase-3-advanced-patterns/3.4-advanced-rag.md) · [Iterative Processing](modules/phase-3-advanced-patterns/3.5-iterative-processing.md) · [FastAPI](modules/phase-3-advanced-patterns/3.6-fastapi-basics.md) · [Clustering & Topics](modules/phase-3-advanced-patterns/3.7-clustering-topics.md) · [Evaluation Systems](modules/phase-3-advanced-patterns/3.8-evaluation-systems.md) · [Document Processing](modules/phase-3-advanced-patterns/3.9-document-processing.md) |
| **[Phase 4: Production & Operations](modules/phase-4-production/0.0-phase-4-production-index.md)**   | Taking agents to production                    | [Docker & Containerization](modules/phase-4-production/4.1-docker.md) · [PostgreSQL + pgvector](modules/phase-4-production/4.2-postgresql-pgvector.md) · [Observability](modules/phase-4-production/4.3-observability.md) · [Guardrails](modules/phase-4-production/4.4-guardrails.md) · [Async & Background Jobs](modules/phase-4-production/4.5-async-background-jobs.md) · [MCP Servers](modules/phase-4-production/4.6-mcp-servers.md) · [Cloud Deployment](modules/phase-4-production/4.7-cloud-deployment.md) · [CI/CD Basics](modules/phase-4-production/4.8-cicd.md) |
| **[Phase 5: Specialization](modules/phase-5-specialization/0.0-phase-5-specialization-index.md)**    | Advanced topics for specific use cases         | [Fine-tuning LLMs](modules/phase-5-specialization/5.1-fine-tuning.md) · [Custom Embeddings](modules/phase-5-specialization/5.2-custom-embeddings.md) · [Advanced NLP](modules/phase-5-specialization/5.3-advanced-nlp.md) · [Multimodal](modules/phase-5-specialization/5.4-multimodal.md) |

## Quick Start


Install dependencies:

```bash
git clone https://github.com/karaoglusina/AI-agent-building-blocks.git
cd AI-agent-building-blocks
uv sync --group phase-1  # or --all-groups for everything (several GB: torch, transformers, faiss, bertopic)
source .venv/bin/activate
```

That's enough for Phase 1. From Phase 2 onwards you also need the spaCy and NLTK data:

```bash
uv sync --group phase-2       # phase-2 … phase-5, or --all-groups
python utils/setup_models.py  # downloads the spaCy model + NLTK corpora
```

### Run without an API key

Most scripts have a `TEST_MODE` that skips the network call and prints what the script *would* do, so you can read and run them before deciding to spend anything:

```bash
TEST_MODE=1 python "scripts/phase-1-foundations/1.1-openai-basics/01_basic_call.py"
```

`TEST_MODE=1` skips API calls, not imports — you still need the dependencies for that phase installed, but you don't need a key.

### Run for real

```bash
# Create .env file
touch .env
# Add your OpenAI API key (replace with your actual key)
echo "OPENAI_API_KEY=sk-..." >> .env
```
Or manually create `.env` and add:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Then:

```bash
python "scripts/phase-1-foundations/1.1-openai-basics/01_basic_call.py"
```

You are ready to explore, check out `modules/` for conceptual walkthroughs and design explanations, structured by phases and modules. You'll find links to related scripts.

### Check everything runs

```bash
python tests/test_all_scripts.py --filter phase-1   # one phase
python tests/test_all_scripts.py                    # all 208, needs --all-groups installed
```

This is a smoke test: it runs each script under `TEST_MODE=1` and reports what executed, not whether the output is right.

## Repository Structure

```
AI-agent-building-blocks/
├── modules/                         # Conceptual walkthroughs (docs) - each document links to relevant scripts - Start here!
│   ├── index.md
│   ├── overview.md                  # The landscape, if you want the map before the code
│   ├── phase-1-foundations/
│   │   ├── 0.0-phase-1-foundations-index.md
│   │   └── 1.1-openai-basics.md
│   │   └── ...
│   └── ...
├── scripts/                         # Runnable Python scripts (one concept per file).
│   ├── phase-1-foundations/
│   │   └── 1.1-openai-basics/01_basic_call.py
│   │   └── ...
│   └── ...
├── utils/                           # Shared helpers (data loading, env, Phase 4 DB models)
├── tests/
│   └── test_all_scripts.py          # Runs every script under TEST_MODE=1
└── data/                            # Sample dataset
    └── sample_job_data.json
```

## The Example Project

Throughout the docs, a single running agent example is used: **AI job market analyzer** that;

- Finds similar roles using embeddings
- Answers questions via RAG without context stuffing
- Classifies positions (ML engineer, researcher, etc.)
- Autonomously researches and synthesizes insights
- Remembers your preferences across sessions

The dataset (`data/sample_job_data.json`) contains 1,318 real LinkedIn postings (14 MB) — messy, diverse, and realistic. It's committed directly to the repo, so a plain `git clone` gets it; no Git LFS or download step needed.

## Scope & Limitations

What this is: a map of the components, so you know what exists and roughly what it's for before you commit to an architecture. What it isn't, and where it stops:

- **It maps production; it doesn't teach deployment.** Phase 2 builds a RAG pipeline over the job dataset. Phase 4 shows PostgreSQL, Docker and observability — but as separate demos, not as your Phase 2 pipeline being taken to production. The failure modes that actually bite on deploy (connection pooling, cold starts, reindexing, retries, tail latency) aren't covered.
- **The vector store choice is made for you, and there's no migration path.** Phases 1-3 use ChromaDB; Phase 4 uses pgvector. There's no shared retrieval interface and no export/import between them, so moving from one to the other means rebuilding rather than swapping an adapter.
- **Observability here is instrumentation, not debugging.** Module 4.3 shows you how to emit traces, costs and metrics. It doesn't teach you how to work out *why* a call produced a bad answer — no trace replay, no diffing retrieved context against expectation.
- **Evaluation is split awkwardly.** Module 2.9, where you meet evaluation first, computes metrics over hardcoded arrays. The part that actually matters — building eval datasets, rubrics, inter-annotator agreement — is in 3.8. If you only read 2.9 you'd reasonably conclude the repo assumes you already have a dataset.
- **Written by someone moving from research into engineering, not from production experience.** The reading is real and cross-referenced against the books below; the scars aren't.

If any of these are what you came for, this repo will point you at the right vocabulary and then leave you to it.

## Tech Stack

**Core:**

- `openai` - OpenAI API client
- `pydantic` - Data validation and structured output
- `chromadb` - Vector database for embeddings
- `numpy` - Vector operations and numerical computing

**NLP & Text Processing:**

- `spacy` - NER, POS tagging, lemmatization, dependency parsing
- `nltk` - Tokenization, stopwords, sentence segmentation
- `keybert` - Embedding-based keyword extraction
- `tiktoken` - Token counting and chunking
- `rapidfuzz` - Fuzzy string matching and similarity

**Production & Infrastructure:**

- `fastapi` - Modern web framework for APIs
- `uvicorn` - ASGI server
- `sqlalchemy` - ORM and database toolkit
- `pgvector` - PostgreSQL extension for vector similarity search
- `celery` - Distributed task queue for background jobs
- `langfuse` - LLM observability and tracing
- `httpx` - Async HTTP client

**ML:**

- `bertopic` - Topic modeling and clustering
- `sentence-transformers` - Sentence embeddings
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-efficient fine-tuning (LoRA)
- `torch` - PyTorch for deep learning
- `sklearn` - Machine learning utilities
- `faiss` - Efficient similarity search (Facebook AI)
- `clip` - Multimodal embeddings (OpenAI CLIP)
- `PIL` (Pillow) - Image processing

## References

The scope and module structure is mainly informed by these great books. I referenced some of the related chapters from the modules.

- Alammar, J., & Grootendorst, M. (2024). *Hands-on large language models: Language understanding and generation*. O'Reilly Media.
- Antić, Z., & Chakravarty, S. (2024). *Python natural language processing cookbook: Over 60 recipes for building powerful NLP solutions using Python and LLM libraries* (2nd ed.). Packt Publishing.
- Huyen, C. (2025). *AI engineering: Building applications with foundation models*. O'Reilly Media.
- Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition* (3rd ed. draft). Stanford University.

Among many YouTube videos and blog posts, [Dave Ebbelaar's channel](https://www.youtube.com/@daveebbelaar) significantly influenced the tool and framework choices. 

## License

MIT   