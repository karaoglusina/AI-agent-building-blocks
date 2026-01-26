"""
03 - Source Citation
====================
Include references in generated answers.

Key concept: Citations let users verify information and build trust.

Book reference: AI_eng.6, hands_on_LLM.II.8
"""

from openai import OpenAI
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import sys
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
from utils.data_loader import load_sample_jobs

client = OpenAI()
openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
chroma = chromadb.Client()
collection = chroma.create_collection(name="jobs_cited", embedding_function=openai_ef)


class Citation(BaseModel):
    """A single citation."""
    source_id: int
    quote: str


class AnswerWithCitations(BaseModel):
    """Answer with source citations."""
    answer: str
    citations: list[Citation]


def index_jobs(jobs: list[dict]):
    """Index jobs with IDs for citation."""
    for i, job in enumerate(jobs, 1):
        collection.add(
            documents=[f"{job['title']} at {job['companyName']}\n{job['description'][:400]}"],
            ids=[str(i)],
            metadatas=[{"source_id": i, "title": job["title"], "company": job["companyName"]}]
        )


def retrieve_with_sources(query: str, n_results: int = 3) -> list[dict]:
    """Retrieve documents with source IDs."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"source_id": meta["source_id"], "title": meta["title"], 
         "company": meta["company"], "text": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def generate_with_citations(query: str, docs: list[dict]) -> AnswerWithCitations:
    """Generate answer with source citations."""
    # Format context with source IDs
    context = "\n\n".join([
        f"[Source {d['source_id']}] {d['title']} at {d['company']}\n{d['text']}"
        for d in docs
    ])
    
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "Answer the question using the provided sources. "
                           "Include citations with source IDs and relevant quotes."
            },
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"}
        ],
        text_format=AnswerWithCitations
    )
    return response.output_parsed


if __name__ == "__main__":
    # Index jobs
    jobs = load_sample_jobs(30)
    index_jobs(jobs)
    print(f"Indexed {collection.count()} jobs\n")
    
    # Query with citations
    query = "What skills are required for Python developer jobs?"
    
    print(f"Q: {query}\n")
    
    docs = retrieve_with_sources(query)
    result = generate_with_citations(query, docs)
    
    print("=== ANSWER ===")
    print(result.answer)
    
    print("\n=== CITATIONS ===")
    for cite in result.citations:
        print(f"[{cite.source_id}] \"{cite.quote}\"")
    
    print("\n=== SOURCES ===")
    for doc in docs:
        print(f"[{doc['source_id']}] {doc['title']} at {doc['company']}")
