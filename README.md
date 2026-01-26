# AI Agent Engineering Curriculum

A hands-on, from-scratch curriculum for learning AI agent development through ~170 focused Python scripts. No frameworks like LangChain or LlamaIndex - just pure Python and lightweight libraries to understand how things actually work.

## 🎯 Philosophy

- **Pure Python + Lightweight Libraries**: Understand the internals, no black boxes
- **Atomic Learning**: One concept per script (~40-80 lines each)
- **Practical Focus**: Real-world patterns, not academic theory
- **Self-Contained**: Each script runs independently
- **Progressive Complexity**: From API basics to production deployment

## 📚 Curriculum Overview

| Phase | Focus | Modules | Status |
|-------|-------|---------|--------|
| **Phase 1** | Foundations | 6 modules, 43 scripts | ✅ Complete |
| **Phase 2** | Core AI + Agents | 9 modules, 65 scripts | ✅ Complete |
| **Phase 3** | Advanced Patterns | 9 modules, ~50 scripts | 🚧 In Progress |
| **Phase 4** | Production | 8 modules, ~35 scripts | 📋 Planned |
| **Phase 5** | Specialization | 4 modules, ~20 scripts | 📋 Planned |

### Phase 1: Foundations (✅ Complete)

- **1.1 OpenAI Basics**: API calls, parameters, streaming, error handling
- **1.2 Pydantic Basics**: Data models, validation, serialization
- **1.3 Structured Output**: Schema-based extraction and classification
- **1.4 Conversations**: Multi-turn chat, context management
- **1.5 Embeddings**: Vector representations, similarity search
- **1.6 Vector Search**: ChromaDB, metadata filtering, hybrid search

### Phase 2: Core AI Engineering (✅ Complete)

- **2.1 Text Preparation**: Cleaning, tokenization, chunking strategies
- **2.2 Information Extraction**: NER, POS tagging, keywords, regex patterns
- **2.3 Classification & Routing**: Zero-shot, few-shot, intent detection
- **2.4 RAG Pipeline**: Retrieval, context assembly, source citation
- **2.5 Agent Orchestration**: Tool calling loops, sequential/parallel execution
- **2.6 Context Engineering**: Token management, prompt assembly, prioritization
- **2.7 Memory Patterns**: Conversation buffers, sliding windows, entity memory
- **2.8 Prompt Engineering**: Chain-of-thought, few-shot, defensive prompting
- **2.9 Evaluation Basics**: Classification/retrieval/generation metrics, LLM-as-judge

### Phase 3: Advanced Patterns (🚧 In Progress)

- **3.1 Advanced Agent Patterns**: ReAct, reflection, planning, self-correction
- **3.2 Multi-Agent Systems**: Agent handoff, specialized agents, coordination
- **3.3 Advanced Memory**: Structured memory, importance scoring, consolidation
- **3.4 Advanced RAG**: Query rewriting, multi-hop, self-RAG, hybrid search
- **3.5 Iterative Processing**: Map-reduce, progressive summarization, batch processing
- **3.6 FastAPI Basics**: API endpoints, streaming, validation
- **3.7 Clustering & Topics**: K-means, UMAP, BERTopic, LLM labeling
- **3.8 Evaluation Systems**: Eval datasets, pipelines, regression testing
- **3.9 Document Processing**: PDF parsing, structured extraction

### Phase 4: Production & Operations (📋 Planned)

Docker, PostgreSQL + pgvector, observability (Langfuse), guardrails, async patterns, background jobs, MCP servers, cloud deployment, CI/CD

### Phase 5: Specialization (📋 Planned)

Fine-tuning, custom embeddings, advanced NLP, multimodal (vision, CLIP)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/karaoglusina/ai-agent-curriculum.git
cd ai-agent-curriculum

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Set up environment variables
export OPENAI_API_KEY="sk-..."
```

### Data Setup

This curriculum uses a dataset of job postings for practical examples. See [`data/README.md`](data/README.md) for setup instructions.

### Run Your First Script

```bash
python modules/phase1/1.1-openai-basics/01_basic_call.py
```

## 📖 Learning Path

### For Beginners
1. Start with Phase 1 modules in order (1.1 → 1.6)
2. Move to Phase 2, focusing on areas that interest you
3. Build a simple agent combining what you've learned

### For Intermediate Developers
- Jump to Phase 2 if you know OpenAI basics
- Focus on agent orchestration (2.5) and RAG (2.4)
- Explore evaluation (2.9) to measure your systems

### For Advanced Engineers
- Review Phases 1-2 for reference patterns
- Implement Phase 3 advanced patterns
- Adapt Phase 4 production patterns to your stack

## 🛠️ Technology Stack

**Core Libraries:**
- `openai` - OpenAI API client
- `pydantic` - Data validation
- `chromadb` - Vector database
- `numpy` - Vector operations

**NLP & Text:**
- `spacy` - NER, POS, lemmatization
- `nltk` - Tokenization, stopwords
- `keybert` - Keyword extraction
- `tiktoken` - Token counting

**Future Phases:**
- FastAPI, SQLAlchemy, Celery, Langfuse, BERTopic, and more

See [`pyproject.toml`](pyproject.toml) for complete dependency list.

## 📁 Project Structure

```
├── data/                    # Dataset (job postings)
├── modules/
│   ├── phase1/             # Foundations (6 modules)
│   ├── phase2/             # Core AI + Agents (9 modules)
│   ├── phase3/             # Advanced patterns (in progress)
│   ├── phase4/             # Production (planned)
│   └── phase5/             # Specialization (planned)
└── utils/
    ├── data_loader.py      # Data loading utilities
    └── models.py           # Pydantic models
```

## 🎓 Script Structure

Each script follows a consistent pattern:

```python
"""
Script Title
============
Brief description of the concept.

Key concept: The main takeaway in one sentence.

Book reference: [relevant books/chapters]
"""

# Clear imports
from openai import OpenAI

def demonstrate_concept():
    """Implementation with type hints and docstrings."""
    pass

if __name__ == "__main__":
    # Runnable example with print statements
    demonstrate_concept()
```

## 🤝 Contributing

This is a personal learning project, but suggestions and improvements are welcome! Feel free to:

- Open issues for bugs or unclear explanations
- Suggest new atomic patterns to add
- Share how you've used these patterns in your projects

## 📚 Book References

The curriculum draws concepts from:

- **AI Engineering** by Chip Huyen (`AI_eng`)
- **Hands-On Large Language Models** by Jay Alammar & Maarten Grotendorst (`hands_on_LLM`)
- **Speech and Language Processing** by Daniel Jurafsky & James H. Martin (`speach_lang`)
- **Python NLP Cookbook** by Zhenya Antic (`NLP_cook`)

## 📝 License

MIT License - feel free to use these patterns in your own projects.

## 🙏 Acknowledgments

Built as a structured learning path to deeply understand AI agent development from first principles.

---

**Star this repo** if you find it helpful! 🌟
