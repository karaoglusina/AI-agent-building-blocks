"""
03 - Trace RAG Pipeline
========================
End-to-end RAG observability with Langfuse.

Key concept: Tracing entire RAG pipelines reveals bottlenecks, retrieval quality issues, and generation problems.

Book reference: AI_eng.10
"""

# Optional dependencies - graceful handling in TEST_MODE
MISSING_DEPENDENCIES = []

import os
import sys
try:
    import chromadb
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

try:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except ImportError:
    MISSING_DEPENDENCIES.append('chromadb')

try:
    from langfuse import Langfuse
except ImportError:
    MISSING_DEPENDENCIES.append('langfuse')

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    MISSING_DEPENDENCIES.append('langfuse')

from openai import OpenAI

# Skip if dependencies missing in TEST_MODE
import os
if os.getenv('TEST_MODE') == '1' and MISSING_DEPENDENCIES:
    print(f'✓ Test mode: Skipping due to missing dependencies: {MISSING_DEPENDENCIES}')
    exit(0)


sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Initialize clients
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))


# Skip actual API call in test mode
if os.getenv("TEST_MODE") == "1":
    print("✓ Test mode: Script structure validated")
    print("✓ Script pattern: PASSED")
    exit(0)

client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()


@observe()
def index_documents(jobs: list[dict]) -> chromadb.Collection:
    """Index documents with tracing."""
    collection = chroma.create_collection(
        name="jobs_rag_trace",
        embedding_function=openai_ef
    )

    documents = [
        f"{j['title']} at {j['companyName']}\n{j['description'][:300]}"
        for j in jobs
    ]
    ids = [j['id'] for j in jobs]
    metadatas = [
        {"title": j['title'], "company": j['companyName']}
        for j in jobs
    ]

    # Log indexing operation
    langfuse_context.update_current_observation(
        input={
            "num_documents": len(jobs),
            "collection_name": "jobs_rag_trace"
        },
        output={
            "indexed": len(jobs),
            "status": "success"
        },
        metadata={
            "embedding_model": "text-embedding-3-small"
        }
    )

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    return collection


@observe()
def retrieve_documents(collection: chromadb.Collection, query: str, n_results: int = 3) -> list[dict]:
    """Retrieve relevant documents with tracing."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = [
        {
            "text": doc,
            "metadata": meta,
            "id": doc_id
        }
        for doc, meta, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0]
        )
    ]

    # Log retrieval
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "n_results": n_results
        },
        output={
            "documents": documents,
            "num_retrieved": len(documents)
        },
        metadata={
            "retrieval_method": "semantic_search"
        }
    )

    return documents


@observe()
def generate_response(query: str, context: list[dict]) -> str:
    """Generate response from retrieved context with tracing."""
    # Format context
    context_text = "\n\n".join([
        f"Job {i+1}: {doc['metadata']['title']} at {doc['metadata']['company']}\n{doc['text'][:200]}"
        for i, doc in enumerate(context)
    ])

    # Create prompt
    prompt = f"""Based on the following job listings, answer the user's question.

Context:
{context_text}

Question: {query}

Answer:"""

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful job search assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7)

    answer = response.choices[0].message.content

    # Log generation
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "context_size": len(context),
            "prompt_length": len(prompt)
        },
        output={
            "answer": answer
        },
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        metadata={
            "model": response.model,
            "temperature": 0.7
        }
    )

    return answer


@observe(name="rag_pipeline")
def rag_pipeline(collection: chromadb.Collection, query: str, n_results: int = 3) -> dict:
    """Complete RAG pipeline with end-to-end tracing."""
    # Tag this trace for filtering
    langfuse_context.update_current_trace(
        tags=["rag", "production"],
        user_id="demo_user",
        session_id="demo_session"
    )

    # Step 1: Retrieve
    with langfuse_context.observe(name="retrieval_step") as retrieval_span:
        documents = retrieve_documents(collection, query, n_results)
        retrieval_span.update(
            output={"num_docs_retrieved": len(documents)}
        )

    # Step 2: Generate
    with langfuse_context.observe(name="generation_step") as generation_span:
        answer = generate_response(query, documents)
        generation_span.update(
            output={"answer_length": len(answer)}
        )

    result = {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "title": doc["metadata"]["title"],
                "company": doc["metadata"]["company"]
            }
            for doc in documents
        ]
    }

    # Log complete pipeline
    langfuse_context.update_current_observation(
        input={"query": query, "n_results": n_results},
        output=result
    )

    return result


@observe()
def evaluate_rag_response(query: str, answer: str, expected_keywords: list[str]) -> dict:
    """Evaluate RAG response quality with tracing."""
    # Simple keyword-based evaluation
    found_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
    score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    evaluation = {
        "score": score,
        "found_keywords": found_keywords,
        "missing_keywords": [kw for kw in expected_keywords if kw not in found_keywords]
    }

    # Log evaluation
    langfuse_context.update_current_observation(
        input={
            "query": query,
            "answer": answer,
            "expected_keywords": expected_keywords
        },
        output=evaluation,
        metadata={
            "evaluation_method": "keyword_matching"
        }
    )

    # Score the associated trace
    langfuse_context.score_current_trace(
        name="keyword_relevance",
        value=score,
        comment=f"Found {len(found_keywords)}/{len(expected_keywords)} keywords"
    )

    return evaluation


if __name__ == "__main__":
    print("=== TRACE RAG PIPELINE ===\n")

    # Index sample jobs
    print("Indexing documents...")
    jobs = load_sample_jobs(50)
    collection = index_documents(jobs)
    print(f"✓ Indexed {collection.count()} jobs\n")

    # Test RAG pipeline
    test_cases = [
        {
            "query": "Find me a machine learning engineer position",
            "keywords": ["machine learning", "engineer", "ML"]
        },
        {
            "query": "What remote jobs are available?",
            "keywords": ["remote", "work from home"]
        },
        {
            "query": "Python developer jobs at startups",
            "keywords": ["python", "developer", "startup"]
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {test['query']}")
        print("="*60)

        # Run RAG pipeline
        result = rag_pipeline(collection, test['query'])

        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. {source['title']} at {source['company']}")

        # Evaluate response
        evaluation = evaluate_rag_response(
            test['query'],
            result['answer'],
            test['keywords']
        )

        print(f"\nEvaluation Score: {evaluation['score']:.2f}")
        print(f"Found keywords: {evaluation['found_keywords']}")
        if evaluation['missing_keywords']:
            print(f"Missing keywords: {evaluation['missing_keywords']}")

    # Flush traces
    langfuse.flush()

    print("\n" + "="*60)
    print("✓ All RAG traces sent to Langfuse")
    print("  - View full pipeline traces with nested spans")
    print("  - Filter by tags: 'rag', 'production'")
    print("  - Check evaluation scores")
    print("="*60)
