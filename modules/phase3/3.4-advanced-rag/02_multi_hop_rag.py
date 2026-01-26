"""
02 - Multi-Hop RAG
==================
Multiple retrieval steps for complex questions.

Key concept: Multi-hop RAG performs iterative retrieval when questions require multiple pieces of information.

Book reference: AI_eng.6, speach_lang.II.14
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


def decompose_question(query: str) -> list[str]:
    """Break complex question into simpler sub-questions."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Break down complex job search questions into 2-3 simpler sub-questions. "
                           "Return ONLY the sub-questions, one per line, no numbering."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    sub_questions = [q.strip() for q in response.output_text.strip().split('\n') if q.strip()]
    return sub_questions


def retrieve_for_subquestion(question: str, n_results: int = 2) -> list[dict]:
    """Retrieve documents for a single sub-question."""
    results = collection.query(query_texts=[question], n_results=n_results)
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def multi_hop_rag(query: str) -> str:
    """Perform multi-hop RAG: decompose → retrieve each → synthesize."""
    print(f"Original Question: {query}\n")

    # Step 1: Decompose into sub-questions
    sub_questions = decompose_question(query)
    print(f"Sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    print()

    # Step 2: Retrieve for each sub-question
    all_context = []
    for i, sq in enumerate(sub_questions, 1):
        print(f"Retrieving for sub-question {i}...")
        docs = retrieve_for_subquestion(sq)
        all_context.extend(docs)

    # Remove duplicates by ID
    unique_context = {doc["text"]: doc for doc in all_context}.values()
    print(f"Retrieved {len(unique_context)} unique documents\n")

    # Step 3: Synthesize final answer
    context_text = "\n\n---\n\n".join([doc["text"] for doc in unique_context])
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Answer the question using the provided context. "
                           "Synthesize information from multiple sources if needed."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}"
            }
        ]
    )
    return response.output_text


if __name__ == "__main__":
    # Index sample jobs
    jobs = load_sample_jobs(50)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")

    # Complex questions requiring multiple hops
    complex_queries = [
        "Compare Python and Java developer salaries and required experience",
        "What skills overlap between data science and ML engineer roles?",
    ]

    for query in complex_queries:
        print("=" * 70)
        answer = multi_hop_rag(query)
        print(f"Final Answer:\n{answer}\n")
