# RAG Pipeline

Retrieval-Augmented Generation is probably the most useful pattern in AI engineering. Instead of hoping the LLM knows the answer, you find relevant information first and include it in the prompt. The LLM generates based on what you retrieved.

## Why This Matters

LLMs have knowledge cutoffs and don't know about your specific data. You can't fit 10,000 job postings in a single prompt. RAG solves both problems: retrieve the relevant pieces, include them in context, generate a grounded answer.

For our job market analyzer, RAG lets you answer questions like "What Python jobs are available in Amsterdam?" by retrieving relevant postings and having the LLM synthesize an answer.

## The Key Ideas

### The Basic Pattern

```
Query → Retrieve → Augment → Generate → Answer
         ↓           ↓          ↓
      Vector DB   Prompt    LLM Call
```

1. **Retrieve**: Find relevant documents using embedding similarity
2. **Augment**: Add retrieved documents to the prompt
3. **Generate**: Ask the LLM to answer based on the context

```python
# Retrieve
results = collection.query(query_texts=[user_query], n_results=5)
context = "\n".join(results["documents"][0])

# Augment + Generate
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based on the following context:"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
)
```

That's RAG in its simplest form.

### Context Assembly

How you present retrieved documents matters. Structure affects quality:

```python
context = """
The following job postings are relevant to your query:

[Job 1: Senior ML Engineer at TechCorp]
- Location: Amsterdam
- Requirements: Python, TensorFlow, 5+ years experience
- Salary: €80,000-100,000

[Job 2: ML Engineer at StartupX]
- Location: Remote (Europe)
- Requirements: Python, PyTorch, 3+ years experience
...

Based on these postings, please answer: {user_query}
"""
```

Clear structure helps the LLM parse and use the context effectively.

### Source Citation

For trust and verification, cite your sources:

```python
class Answer(BaseModel):
    response: str
    sources: list[str]  # IDs of documents used
    confidence: Literal["high", "medium", "low"]

# Prompt the model to cite which documents it used
```

Users can verify the answer against the original documents.

### Handling No Results

What happens when retrieval returns nothing relevant?

```python
if not results or results["distances"][0][0] > threshold:
    return "I couldn't find relevant information. Could you rephrase your question?"
```

Don't force the LLM to answer from nothing. Detect low-quality retrieval and handle it gracefully.

### Metadata Filtering

Combine semantic search with structured filters:

```python
results = collection.query(
    query_texts=["machine learning engineer"],
    n_results=10,
    where={"location": "remote", "salary_min": {"$gte": 100000}}
)
```

Filter first (remote, high salary), then rank by semantic relevance (ML engineer).

## What's in This Module

| Script | What it shows |
|--------|---------------|
| 01_basic_rag.py | End-to-end RAG pipeline |
| 02_context_assembly.py | Formatting retrieved context |
| 03_source_citation.py | Including references in answers |
| 04_no_results_handling.py | Graceful fallbacks |
| 05_rag_with_filters.py | Combining semantic search with filters |
| 06_rag_evaluation.py | Measuring RAG quality |

## Evaluation Metrics

| Metric | What it measures |
|--------|------------------|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant docs found in top-K |
| MRR | How high is the first relevant result? |
| NDCG | Overall ranking quality |

Evaluate retrieval separately from generation. Bad retrieval → bad answers.

## Things to Think About

- **How many documents should you retrieve?** Too few = might miss relevant info. Too many = noise drowns signal. 3-10 is typical.
- **What if documents contradict each other?** The LLM might get confused. Consider filtering for consistency or asking the model to note contradictions.
- **When does RAG fail?** When the question requires reasoning across many documents. When the relevant information wasn't indexed. When retrieval returns similar-but-wrong results.

## Related

- [Vector Search](../phase-1-foundations/vector-search.md) - The retrieval foundation
- [Context Engineering](./context-engineering.md) - Managing context limits
- [Agent Orchestration](./agent-orchestration.md) - RAG as an agent tool
- [Advanced RAG](../phase-3-advanced-patterns/advanced-rag.md) - More sophisticated retrieval

## Book References

- AI_eng.6 - RAG architecture
- hands_on_LLM.II.8 - Retrieval systems
- speach_lang.II.14 - Question answering
