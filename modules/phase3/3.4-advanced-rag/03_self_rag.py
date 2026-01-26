"""
03 - Self-RAG
=============
Decide when retrieval is needed.

Key concept: Self-RAG lets the LLM decide whether to retrieve, avoiding unnecessary lookups.

Book reference: AI_eng.6
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])
from utils.data_loader import load_sample_jobs

# Setup
client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="jobs", embedding_function=openai_ef)


def index_jobs(jobs: list[dict]):
    """Index jobs into vector database."""
    documents = [f"{j['title']} at {j['companyName']}\n{j['description'][:500]}" for j in jobs]
    ids = [j['id'] for j in jobs]
    metadatas = [{"title": j['title'], "company": j['companyName']} for j in jobs]
    collection.add(documents=documents, ids=ids, metadatas=metadatas)


def should_retrieve(query: str) -> bool:
    """Decide if retrieval is needed for this query."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are helping with a job search system. "
                           "Decide if the question requires looking up job listings (answer 'RETRIEVE') "
                           "or can be answered directly (answer 'DIRECT'). "
                           "Return ONLY 'RETRIEVE' or 'DIRECT', nothing else."
            },
            {
                "role": "user",
                "content": f"Question: {query}"
            }
        ]
    )
    decision = response.output_text.strip().upper()
    return decision == "RETRIEVE"


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """Retrieve relevant documents."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def generate_with_context(query: str, context: list[dict]) -> str:
    """Generate answer using retrieved context."""
    context_text = "\n\n---\n\n".join([doc["text"] for doc in context])
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Answer questions about jobs using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]
    )
    return response.output_text


def generate_direct(query: str) -> str:
    """Generate answer without retrieval."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question directly."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    return response.output_text


def self_rag(query: str) -> tuple[str, bool]:
    """Self-RAG: decide if retrieval is needed, then answer."""
    needs_retrieval = should_retrieve(query)
    print(f"Query: {query}")
    print(f"Decision: {'RETRIEVE' if needs_retrieval else 'DIRECT'}\n")

    if needs_retrieval:
        context = retrieve(query)
        answer = generate_with_context(query, context)
    else:
        answer = generate_direct(query)

    return answer, needs_retrieval


if __name__ == "__main__":
    # Index sample jobs
    jobs = load_sample_jobs(50)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")

    # Mix of questions - some need retrieval, some don't
    test_queries = [
        "What Python jobs are available?",  # Needs retrieval
        "What does RAG stand for?",  # Direct answer
        "Show me data science positions in Amsterdam",  # Needs retrieval
        "How do I write a good resume?",  # Direct answer
        "Are there any remote backend roles?",  # Needs retrieval
    ]

    for query in test_queries:
        print("=" * 60)
        answer, used_retrieval = self_rag(query)
        print(f"Answer: {answer}\n")
