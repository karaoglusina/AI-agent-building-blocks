# 🎓 AI Agent Fundamentals Curriculum

A progressive learning path from OpenAI basics to building intelligent agents that work with textual data.

## 📚 Lessons Overview

| Lesson | Topic | Scripts | Focus |
|--------|-------|---------|-------|
| **01** | OpenAI Basics | 8 | API calls, parameters, streaming |
| **02** | Pydantic Basics | 8 | Data models, validation, serialization |
| **03** | Structured Output | 7 | Extract & classify with schemas |
| **04** | Conversations | 6 | Multi-turn, context management |
| **05** | Embeddings | 8 | Vector representations, similarity |
| **06** | Vector Search | 6 | ChromaDB, metadata filtering |

**Total: 43 short, focused scripts**

## 🚀 Quick Start

```bash
# Install dependencies
pip install openai pydantic chromadb numpy

# Set your API key
export OPENAI_API_KEY="your-key-here"

# Start with Lesson 01
cd lessons/01-openai-basics
python 01_basic_call.py
```

## 📖 Recommended Learning Path

### Phase 1: Foundations (Do First)
1. `01-openai-basics/` - Understand the API
2. `02-pydantic-basics/` - Data validation
3. `03-structured-output/` - The magic of schema-based extraction

### Phase 2: Conversational AI
4. `04-conversations/` - Build chat capabilities

### Phase 3: Semantic Search
5. `05-embeddings/` - Vector representations
6. `06-vector-search/` - ChromaDB for production

### Phase 4: Integration (Coming Soon)
- Tool-augmented agents
- RAG pipelines
- Job market chatbot

## 🏗️ Project Structure

```
sina-agent-testing/
├── data/
│   └── job_post_data.json     # 10k+ job postings
├── lessons/
│   ├── 01-openai-basics/      # API fundamentals
│   ├── 02-pydantic-basics/    # Data models
│   ├── 03-structured-output/  # Schema-based extraction
│   ├── 04-conversations/      # Chat capabilities
│   ├── 05-embeddings/         # Vector representations
│   └── 06-vector-search/      # ChromaDB
├── utils/
│   ├── data_loader.py         # Load job data
│   └── models.py              # Shared Pydantic models
└── building-blocks/           # Previous agent components
```

## 💡 Design Philosophy

Each script is:
- **Short** - One concept per file
- **Runnable** - Execute directly with `python script.py`
- **Self-contained** - Minimal dependencies between scripts
- **Practical** - Uses real job data where possible

## 🎯 End Goal

Build a job market chatbot that can:
- Answer questions about the job market
- Search jobs semantically ("find Python roles similar to X")
- Extract and classify job information
- Create dynamic categories on-the-fly
- Combine vector search with conversational context

## 📦 Dependencies

```
openai>=1.0.0
pydantic>=2.0.0
chromadb>=0.4.0
numpy>=1.24.0
```

## 🔑 Environment

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (for some scripts)
export OPENAI_ORG_ID="org-..."
```

### Setting Environment Variables

**Temporary (current terminal session only):**
```bash
export OPENAI_ORG_ID="org-..."
```

**Permanent (add to your shell config):**
```bash
# Add to ~/.zshrc (or ~/.bashrc for bash)
echo 'export OPENAI_ORG_ID="org-..."' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

**Using a `.env` file (recommended for projects):**
Create a `.env` file in your project root and use a tool like `python-dotenv`:
```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...
```
