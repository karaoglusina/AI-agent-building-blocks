# AI Agent Building Blocks

A collection of atomic Python scripts covering every component you need to build AI agents. Each script demonstrates one concept in isolation - no frameworks, just understanding how things actually work.

I built this repository while learning AI agent development to:

- **Understand the fundamentals** - Writing each pattern from scratch (no LangChain/LlamaIndex) forced me to understand what's actually happening under the hood
- **Create a comprehensive reference** - When I need to implement something, I have working examples of every component
- **Document the landscape** - Before building a system, it helps to know what building blocks exist and how they work in isolation

Think of this as a reference library of patterns; to build something useful, you need to:

- Understand your specific problem
- Design an architecture that fits your use case
- Compose these components thoughtfully
- Add proper error handling, monitoring, testing, etc.

## What's Covered

I shaped the scope and organization of modules by studying the following books. These books helped me understand what components are needed when building AI systems, ensuring the curriculum covers the full stack from basics to production.

- **AI Engineering: Building Applications with Foundation Models** by Chip Huyen (O'Reilly, 2024) - For production patterns and best practices
- **Hands-On Large Language Models: Language Understanding and Generation** by Jay Alammar and Maarten Grootendorst (O'Reilly, 2024) - For practical LLM applications
- **Speech and Language Processing** by Daniel Jurafsky and James H. Martin (3rd edition) - For NLP fundamentals
- **Python Natural Language Processing Cookbook** by Zhenya Antić (Packt Publishing) - For practical NLP patterns

The repository spans the entire stack of AI agent development:

### [Phase 1: Foundations](docs/phase-1-foundations/0.0-phase-1-foundations-index.md)

Learn to work with the OpenAI API and structure your data:

- **[OpenAI API](docs/phase-1-foundations/1.1-openai-basics.md)**: Basic calls, streaming, parameters, error handling
- **[Pydantic Models](docs/phase-1-foundations/1.2-pydantic-basics.md)**: Data validation, nested models, serialization
- **[Structured Output](docs/phase-1-foundations/1.3-structured-output.md)**: Extract structured data using schemas instead of parsing text
- **[Conversations](docs/phase-1-foundations/1.4-conversations.md)**: Multi-turn chat, context windows, memory management
- **[Embeddings](docs/phase-1-foundations/1.5-embeddings.md)**: Vector representations, similarity search, semantic retrieval
- **[Vector Databases](docs/phase-1-foundations/1.6-vector-search.md)**: ChromaDB basics, metadata filtering, CRUD operations

### [Phase 2: Core AI Engineering](docs/phase-2-building-ai-systems/0.0-phase-2-building-ai-systems-index.md)

The patterns you'll use daily when building AI systems:

- **[Text Preparation](docs/phase-2-building-ai-systems/2.1-text-preparation.md)**: Cleaning, tokenization, chunking strategies (fixed/semantic/recursive)
- **[Information Extraction](docs/phase-2-building-ai-systems/2.2-information-extraction.md)**: NER, POS tagging, keyword extraction (TF-IDF, KeyBERT), regex patterns, fuzzy matching
- **[Classification & Routing](docs/phase-2-building-ai-systems/2.3-classification-routing.md)**: Zero-shot, few-shot, intent detection, query routing, sentiment analysis
- **[RAG Pipeline](docs/phase-2-building-ai-systems/2.4-rag-pipeline.md)**: Retrieval, context assembly, source citation, handling no results, metadata filtering
- **[Agent Orchestration](docs/phase-2-building-ai-systems/2.5-agent-orchestration.md)**: Tool calling loops, sequential/parallel execution, error recovery
- **[Context Engineering](docs/phase-2-building-ai-systems/2.6-context-engineering.md)**: Token counting, prompt assembly, context prioritization, compression
- **[Memory Patterns](docs/phase-2-building-ai-systems/2.7-memory-patterns.md)**: Conversation buffers, sliding windows, summary memory, entity tracking
- **[Prompt Engineering](docs/phase-2-building-ai-systems/2.8-prompt-engineering.md)**: Chain-of-thought, few-shot examples, defensive prompting, self-consistency
- **[Evaluation](docs/phase-2-building-ai-systems/2.9-evaluation-basics.md)**: Classification/retrieval/generation metrics, LLM-as-judge, A/B testing

### [Phase 3: Advanced Patterns](docs/phase-3-advanced-patterns/0.0-phase-3-advanced-patterns-index.md)

Sophisticated agent behaviors and production concerns:

- **[Advanced Agents](docs/phase-3-advanced-patterns/3.1-advanced-agent-patterns.md)** (7 scripts): ReAct, reflection, planning, self-correction, confidence routing, iterative refinement, tree-of-thought
- **[Multi-Agent Systems](docs/phase-3-advanced-patterns/3.2-multi-agent-systems.md)** (5 scripts): Agent handoff, specialized agents, coordinator patterns, preference detection, agent communication
- **[Advanced Memory](docs/phase-3-advanced-patterns/3.3-advanced-memory.md)** (5 scripts): Structured memory, importance scoring, memory consolidation, preference systems, episodic memory
- **[Advanced RAG](docs/phase-3-advanced-patterns/3.4-advanced-rag.md)** (6 scripts): Query rewriting, multi-hop retrieval, self-RAG, cross-encoder reranking, hybrid search, RAG fusion
- **[Iterative Processing](docs/phase-3-advanced-patterns/3.5-iterative-processing.md)** (5 scripts): Map-reduce, progressive summarization, refinement chains, hierarchical processing, batch processing
- **[FastAPI](docs/phase-3-advanced-patterns/3.6-fastapi-basics.md)** (6 scripts): Building APIs for your agents (endpoints, async, streaming, chat, RAG APIs)
- **[Clustering & Topics](docs/phase-3-advanced-patterns/3.7-clustering-topics.md)** (6 scripts): K-means, UMAP visualization, BERTopic, LLM-based cluster labeling, topic coherence, interactive exploration
- **[Evaluation Systems](docs/phase-3-advanced-patterns/3.8-evaluation-systems.md)** (6 scripts): Test datasets, evaluation pipelines, regression testing, prompt versioning, cost tracking, human eval design
- **[Document Processing](docs/phase-3-advanced-patterns/3.9-document-processing.md)** (4 scripts): PDF parsing with PyMuPDF/pypdf, structured extraction, full pipeline to chunks

### [Phase 4: Production & Operations](docs/phase-4-production/0.0-phase-4-production-index.md)

Taking agents to production:

- **[Docker & Containerization](docs/phase-4-production/4.1-docker.md)** (5 files): Dockerfiles, docker-compose, environment configuration, multi-service orchestration
- **[PostgreSQL + pgvector](docs/phase-4-production/4.2-postgresql-pgvector.md)** (6 scripts): SQLAlchemy ORM, CRUD operations, Alembic migrations, pgvector setup, vector search, hybrid search
- **[Observability](docs/phase-4-production/4.3-observability.md)** (5 scripts): Langfuse setup, tracing LLM calls, RAG pipeline tracing, cost monitoring, custom metrics
- **[Guardrails](docs/phase-4-production/4.4-guardrails.md)** (8 scripts): Input validation, prompt injection defense, jailbreak defense, PII filtering, output validation, content moderation, architecture, model gateway
- **[Async & Background Jobs](docs/phase-4-production/4.5-async-background-jobs.md)** (5 scripts): Asyncio basics, concurrent LLM calls, Celery setup, background tasks, task status tracking
- **[MCP Servers](docs/phase-4-production/4.6-mcp-servers.md)** (4 scripts): MCP overview, client connection, tool usage, custom server creation
- **[Cloud Deployment](docs/phase-4-production/4.7-cloud-deployment.md)** (4 files): VM setup guide, HTTPS configuration, health checks, logging configuration
- **[CI/CD Basics](docs/phase-4-production/4.8-cicd.md)** (3 files): GitHub Actions workflows for testing, automation, deployment

### [Phase 5: Specialization](docs/phase-5-specialization/0.0-phase-5-specialization-index.md)

Advanced topics for specific use cases:

- **[Fine-tuning LLMs](docs/phase-5-specialization/5.1-fine-tuning.md)** (6 scripts): Decision framework, data preparation, LoRA/QLoRA, quantization, SFT/RLHF/DPO overview, evaluation
- **[Custom Embeddings](docs/phase-5-specialization/5.2-custom-embeddings.md)** (4 scripts): sentence-transformers, domain adaptation with TSDAE, embedding evaluation, bias awareness
- **[Advanced NLP](docs/phase-5-specialization/5.3-advanced-nlp.md)** (3 scripts): Dependency parsing, relation extraction, coreference resolution
- **[Multimodal](docs/phase-5-specialization/5.4-multimodal.md)** (4 scripts): GPT-4V vision basics, CLIP text-image similarity, multimodal search, document vision

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/karaoglusina/agent-building-blocks.git
cd agent-building-blocks

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Set up environment variables
export OPENAI_API_KEY="sk-..."

# Set up data (see data/README.md)
# The scripts use a job postings dataset - you can use your own or create sample data
```

### Run Your First Script

```bash
python modules/phase1/1.1-openai-basics/01_basic_call.py
```

## How to Use This Repository

### As a Learning Path

Start from Phase 1, module 1.1 and work through in order. Each script builds conceptual understanding.

### As a Reference

Need to implement RAG? Check `modules/phase2/2.4-rag-pipeline/`. Need agent orchestration? Look at `modules/phase2/2.5-agent-orchestration/`.

### As Building Blocks

Copy patterns you need into your project and adapt them. These are starting points, not final solutions.

## Example: What a Complete System Looks Like

These scripts show individual components. A real agent might combine:

1. **Text Preparation** (2.1) - Clean and chunk documents
2. **Vector Search** (1.6) - Store and retrieve chunks
3. **RAG Pipeline** (2.4) - Retrieve relevant context
4. **Agent Orchestration** (2.5) - Give the agent tools to search, extract, and respond
5. **Memory Patterns** (2.7) - Remember user preferences across conversations
6. **Context Engineering** (2.6) - Fit everything into the context window

Each piece is simple. The complexity comes from putting them together thoughtfully.

## Technology Stack

**Core:**

- `openai` - OpenAI API client
- `pydantic` - Data validation
- `chromadb` - Vector database
- `numpy` - Vector operations

**NLP & Text:**

- `spacy` - NER, POS, lemmatization
- `nltk` - Tokenization, stopwords
- `keybert` - Keyword extraction
- `tiktoken` - Token counting
- `rapidfuzz` - Fuzzy string matching

**Production:**

- FastAPI, SQLAlchemy, Celery, Langfuse, BERTopic, sentence-transformers, and more

## License

MIT
