# AI Agent Building Blocks

A collection of atomic Python scripts covering every component you need to build AI agents. Each script demonstrates one concept in isolation - no frameworks, just understanding how things actually work.

I built this repository while learning AI agent development to:

- **Understand the fundamentals** - Writing each pattern from scratch (no LangChain/LlamaIndex) forced me to understand what's actually happening under the hood
- **Create a comprehensive reference** - When I need to implement something, I have working examples of every component
- **Document the landscape** - Before building a system, it helps to know what building blocks exist and how they work in isolation

Think of this as a reference library of patterns, not a framework or ready-to-deploy solution. These are building blocks, not complete systems. To build something useful, you need to:

- Understand your specific problem
- Design an architecture that fits your use case
- Compose these components thoughtfully
- Add proper error handling, monitoring, testing, etc.

Each script follows these principles:

- **Atomic**: One concept per file (~40-80 lines)
- **No frameworks**: Pure Python + lightweight libraries (understand the internals)
- **Runnable**: Execute directly with `python script.py`
- **Self-contained**: Minimal dependencies between scripts
- **Practical**: Uses real data (job postings dataset) where applicable
- **Type-hinted**: All functions have type annotations
- **Documented**: Clear docstrings explaining the "why" not just the "what"



## What's Covered

I shaped the scope and organization of modules was by studying the following books. These books helped me understand what components are needed when building AI systems, ensuring the curriculum covers the full stack from basics to production.

- **AI Engineering: Building Applications with Foundation Models** by Chip Huyen (O'Reilly, 2024) - For production patterns and best practices
- **Hands-On Large Language Models: Language Understanding and Generation** by Jay Alammar and Maarten Grootendorst (O'Reilly, 2024) - For practical LLM applications
- **Speech and Language Processing** by Daniel Jurafsky and James H. Martin (3rd edition) - For NLP fundamentals
- **Python Natural Language Processing Cookbook** by Zhenya Antić (Packt Publishing) - For practical NLP patterns

The repository spans the entire stack of AI agent development:

### Phase 1: Foundations (✅ 43 scripts)

Learn to work with the OpenAI API and structure your data:

- **OpenAI API**: Basic calls, streaming, parameters, error handling
- **Pydantic Models**: Data validation, nested models, serialization
- **Structured Output**: Extract structured data using schemas instead of parsing text
- **Conversations**: Multi-turn chat, context windows, memory management
- **Embeddings**: Vector representations, similarity search, semantic retrieval
- **Vector Databases**: ChromaDB basics, metadata filtering, CRUD operations

### Phase 2: Core AI Engineering (✅ 65 scripts)

The patterns you'll use daily when building AI systems:

- **Text Preparation**: Cleaning, tokenization, chunking strategies (fixed/semantic/recursive)
- **Information Extraction**: NER, POS tagging, keyword extraction (TF-IDF, KeyBERT), regex patterns, fuzzy matching
- **Classification & Routing**: Zero-shot, few-shot, intent detection, query routing, sentiment analysis
- **RAG Pipeline**: Retrieval, context assembly, source citation, handling no results, metadata filtering
- **Agent Orchestration**: Tool calling loops, sequential/parallel execution, error recovery
- **Context Engineering**: Token counting, prompt assembly, context prioritization, compression
- **Memory Patterns**: Conversation buffers, sliding windows, summary memory, entity tracking
- **Prompt Engineering**: Chain-of-thought, few-shot examples, defensive prompting, self-consistency
- **Evaluation**: Classification/retrieval/generation metrics, LLM-as-judge, A/B testing

### Phase 3: Advanced Patterns (🚧 In Progress)

Sophisticated agent behaviors and production concerns:

- **Advanced Agents**: ReAct, reflection, planning, self-correction, tree-of-thought
- **Multi-Agent Systems**: Agent handoff, specialized agents, coordinator patterns
- **Advanced Memory**: Structured memory, importance scoring, memory consolidation
- **Advanced RAG**: Query rewriting, multi-hop retrieval, self-RAG, hybrid search (BM25 + semantic)
- **Iterative Processing**: Map-reduce, progressive summarization, batch processing
- **FastAPI**: Building APIs for your agents (endpoints, streaming, validation)
- **Clustering & Topics**: K-means, UMAP visualization, BERTopic, LLM-generated labels
- **Evaluation Systems**: Test datasets, evaluation pipelines, regression testing, cost tracking
- **Document Processing**: PDF parsing, structured extraction from documents

### Phase 4: Production & Operations (📋 Planned)

Taking agents to production:

- Docker containerization
- PostgreSQL + pgvector for production vector storage
- Observability with Langfuse
- Guardrails (input validation, prompt injection defense, PII filtering)
- Async patterns and background jobs (Celery)
- MCP servers for tool integration
- Cloud deployment
- CI/CD basics

### Phase 5: Specialization (📋 Planned)

Advanced topics for specific use cases:

- Fine-tuning LLMs (when and how)
- Custom embeddings and domain adaptation
- Advanced NLP (dependency parsing, relation extraction, coreference)
- 

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

**Coming in Phase 3-5:**

- FastAPI, SQLAlchemy, Celery, Langfuse, BERTopic, sentence-transformers, and more

## License

## License

**Questions or feedback?** Open an issue or reach out. Happy building! 🚀