# Phase 3: Advanced Patterns

Phase 2 gave you the core patterns. Phase 3 makes them more powerful. More sophisticated agent reasoning, multi-agent systems, advanced RAG techniques, and the tools to serve and scale.

## What You're Building Toward

By the end of Phase 3, you'll have:
- Agents that plan, reflect, and correct themselves
- Multi-agent systems that coordinate on complex tasks
- Sophisticated memory systems for personalization
- Advanced RAG with reranking, hybrid search, and query rewriting
- APIs serving your AI systems
- Clustering and topic modeling for exploration
- Production-grade evaluation pipelines
- Document processing for PDFs and complex content

This is where your job market analyzer becomes genuinely useful - capable of complex research, multi-step analysis, and reliable operation.

## The Modules

### [Advanced Agent Patterns](./advanced-agent-patterns.md)
ReAct, planning, self-reflection, tree of thought. Patterns that make agents reason better and catch their own mistakes.

### [Multi-Agent Systems](./multi-agent-systems.md)
When one agent isn't enough. Coordinator patterns, specialized agents, handoffs, and inter-agent communication.

### [Advanced Memory](./advanced-memory.md)
Beyond simple buffers. Structured memory, importance scoring, consolidation, preference tracking, episodic recall.

### [Advanced RAG](./advanced-rag.md)
When basic RAG isn't enough. Query rewriting, multi-hop retrieval, hybrid search, reranking, RAG fusion.

### [Iterative Processing](./iterative-processing.md)
Handling content too large for one LLM call. Map-reduce, progressive summarization, refinement chains.

### [FastAPI Basics](./fastapi-basics.md)
Serving your AI systems as APIs. Request validation, async endpoints, streaming, chat and RAG APIs.

### [Clustering & Topics](./clustering-topics.md)
Discovering structure in your data. K-means, BERTopic, UMAP visualization, automatic labeling.

### [Evaluation Systems](./evaluation-systems.md)
Production-grade evaluation. Test datasets, automated pipelines, regression testing, prompt versioning.

### [Document Processing](./document-processing.md)
Getting text from PDFs. PyMuPDF, structure preservation, chunking strategies, full pipelines.

## The Flow

Phase 3 modules are more independent than earlier phases:

```
Advanced Agent Patterns ←→ Multi-Agent Systems
                              ↓
                       Advanced Memory
                              ↓
                        Advanced RAG
                              ↓
                    Iterative Processing
                              ↓
                      FastAPI Basics → Evaluation Systems
                              ↓
Clustering & Topics ←→ Document Processing
```

Start with what you need. Advanced Agent Patterns and Advanced RAG are probably the most immediately useful.

## What You'll Need

This phase adds more dependencies:

```bash
# APIs
pip install fastapi uvicorn httpx

# Search
pip install sentence-transformers rank-bm25

# Clustering
pip install bertopic umap-learn matplotlib

# Documents
pip install pymupdf pypdf
```

## The Job Market Analyzer at Phase 3

With Phase 3 techniques, you can:
- Use ReAct for multi-step job research tasks
- Have specialized agents for search, analysis, and recommendations
- Remember user preferences across sessions
- Answer complex questions requiring multiple retrieval steps
- Process job description PDFs and add them to your knowledge base
- Discover market segments through clustering
- Serve it all through a proper API
